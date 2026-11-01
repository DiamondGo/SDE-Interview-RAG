import json
import os
from datetime import datetime

import httpx
import pytest

from sde_rag import db
from sde_rag.graphs import ingest as ingest_graph
from sde_rag.http import HttpClient
from sde_rag.parse.clean import fragment_to_markdown
from sde_rag.pipeline.fetch import source_due, sync_source
from sde_rag.sources.base import DocRef, ListResult
from sde_rag.sources.feeds import parse_feed
from sde_rag.sources.leetcode import LeetCodeDiscuss
from sde_rag.sources.medium import Medium, author_feed
from sde_rag.sources.onepoint3acres import HIDDEN_MARK, OnePoint3Acres, parse_when, visible_text
from sde_rag.sources.reddit import Reddit
from tests.conftest import make_registry
from tests.fakes import PREP_MD, FakeConnector


def _http(routes: dict, calls: list | None = None) -> HttpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        key = str(request.url)
        if calls is not None:
            calls.append((request.method, key, request.content.decode() if request.content else ""))
        for prefix, reply in routes.items():
            if key.startswith(prefix):
                body = reply(request) if callable(reply) else reply
                return httpx.Response(200, text=body if isinstance(body, str) else json.dumps(body))
        return httpx.Response(404, text="not found")
    return HttpClient(delay_s=0, transport=httpx.MockTransport(handler))


REDDIT_ATOM = """<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><author><name>/u/alice</name></author><id>t3_abc</id><updated>2026-10-31T10:00:00+00:00</updated>
<title>Google L4 onsite experience</title><link href="https://www.reddit.com/r/leetcode/comments/abc/google/"/>
<content type="html">&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;Round 1: &lt;code&gt;LRU cache&lt;/code&gt;.&lt;/p&gt;
&lt;ul&gt;&lt;li&gt;Round 2: design a rate limiter&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;&amp;#32; submitted by &amp;#32;
&lt;a href="https://www.reddit.com/user/alice"&gt; /u/alice &lt;/a&gt; &lt;span&gt;&lt;a href="https://www.reddit.com/r/leetcode/comments/abc/"&gt;[link]&lt;/a&gt;&lt;/span&gt;</content></entry>
<entry><id>t3_def</id><updated>2026-10-31T11:00:00+00:00</updated><title>My blog post</title>
<link href="https://www.reddit.com/r/leetcode/comments/def/blog/"/>
<content type="html">&lt;a href="https://example.com/my-onsite"&gt;[link]&lt;/a&gt;</content></entry></feed>"""


def test_reddit_reads_posts_from_the_feed_and_keeps_link_targets():
    calls = []
    con = Reddit("reddit", {"subreddits": ["leetcode"]}, _http({"https://www.reddit.com/r/leetcode/search.rss": REDDIT_ATOM}, calls))
    res = con.list_updates({})
    assert [r.external_id for r in res.refs] == ["t3_abc", "t3_def"] and not res.errors
    assert res.refs[0].outbound_url is None and res.refs[1].outbound_url == "https://example.com/my-onsite"
    doc = con.fetch_doc(res.refs[0], None)
    assert len(calls) == 1                                  # the post itself costs no request
    assert "Round 1: `LRU cache`." in doc.content and "- Round 2: design a rate limiter" in doc.content
    assert "submitted by" not in doc.content and doc.origin == "reddit/r/leetcode" and doc.author == "alice"


MEDIUM_TAG = """<?xml version="1.0"?><rss xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0"><channel>
<item><title>My Uber interview</title><link>https://ritwik.medium.com/my-uber-interview-123?source=rss</link>
<guid>https://medium.com/p/123</guid><dc:creator>Ritwik</dc:creator><pubDate>Thu, 29 Oct 2026 07:36:22 GMT</pubDate>
<description>&lt;p&gt;snippet only&lt;/p&gt;</description><category>interview-experience</category></item>
<item><title>Pub post</title><link>https://medium.com/frontend-army/pub-post-456</link><guid>https://medium.com/p/456</guid>
<description>&lt;p&gt;Pub snippet&lt;/p&gt;</description></item></channel></rss>"""
MEDIUM_AUTHOR = """<?xml version="1.0"?><rss xmlns:content="http://purl.org/rss/1.0/modules/content/" version="2.0"><channel>
<item><title>My Uber interview</title><guid>https://medium.com/p/123</guid>
<content:encoded>&lt;h3&gt;Round 1&lt;/h3&gt;&lt;p&gt;Design Uber eats.&lt;/p&gt;&lt;pre&gt;code&lt;br&gt;more&lt;/pre&gt;</content:encoded></item>
</channel></rss>"""


def test_medium_reads_full_text_from_the_author_feed_once_and_falls_back_to_the_snippet():
    calls = []
    con = Medium("medium", {"tags": ["interview-experience"]}, _http({
        "https://medium.com/feed/tag/": MEDIUM_TAG, "https://ritwik.medium.com/feed": MEDIUM_AUTHOR,
        "https://medium.com/feed/frontend-army": "<rss><channel></channel></rss>"}, calls))
    refs = con.list_updates({}).refs
    assert [r.url for r in refs] == ["https://ritwik.medium.com/my-uber-interview-123",
                                     "https://medium.com/frontend-army/pub-post-456"]
    doc = con.fetch_doc(refs[0], None)
    assert "### Round 1" in doc.content and "Design Uber eats." in doc.content and "```\ncode\nmore\n```" in doc.content
    con.fetch_doc(refs[0], None)
    assert sum(1 for c in calls if "ritwik.medium.com/feed" in c[1]) == 1
    assert "Pub snippet" in con.fetch_doc(refs[1], None).content
    assert author_feed("https://medium.com/@bob/a-post-1") == "https://medium.com/feed/@bob"


def _lc_page(nodes):
    return {"data": {"ugcArticleDiscussionArticles": {"edges": [{"node": n} for n in nodes]}}}


def _node(tid, created):
    return {"topicId": tid, "slug": f"post-{tid}", "title": f"Post {tid}", "createdAt": created, "updatedAt": created}


def test_leetcode_lists_newest_first_down_to_the_high_water_mark_and_rewinds_for_failures():
    pages = [[_node(3, "2026-10-31T10:00:00+00:00"), _node(2, "2026-10-30T10:00:00+00:00")],
             [_node(1, "2026-10-20T10:00:00+00:00")], []]

    def graphql(request):
        body = json.loads(request.content)
        if "ugcArticleDiscussionArticle(" in body["query"]:
            return {"data": {"ugcArticleDiscussionArticle": {
                "title": "Post 2", "content": "Asked: two sum.", "createdAt": "2026-10-30T10:00:00+00:00",
                "author": {"userName": "u"}, "tags": [{"slug": "google"}]}}}
        return _lc_page(pages[body["variables"]["skip"] // 2])

    calls = []
    con = LeetCodeDiscuss("leetcode", {"tags": ["interview"], "page_size": 2, "max_pages": 5},
                          _http({"https://leetcode.com/graphql": graphql}, calls))
    res = con.list_updates({"hw:interview": datetime.fromisoformat("2026-10-30T12:00:00+00:00").timestamp()})
    assert [r.external_id for r in res.refs] == ["2", "3"]          # post 1 is older than the overlap window
    assert len(calls) == 2                                  # page 2 is all older: no third page
    assert res.state["hw:interview"] == datetime.fromisoformat("2026-10-31T10:00:00+00:00").timestamp()
    doc = con.fetch_doc(res.refs[0], None)
    assert doc.content.startswith("# Post 2") and "two sum" in doc.content and "Tags: google" in doc.content
    rewound = con.finalize(dict(res.state), [res.refs[0]])
    assert rewound["hw:interview"] == res.refs[0].meta["created_ts"]
    assert all(c[0] == "POST" for c in calls)


def test_leetcode_graphql_errors_are_errors():
    con = LeetCodeDiscuss("leetcode", {"tags": ["interview"]}, _http({"https://leetcode.com/graphql": {
        "errors": [{"message": "rate limited"}]}}))
    res = con.list_updates({})
    assert res.refs == [] and "rate limited" in res.errors[0] and "hw:interview" not in res.state


def test_onepoint3acres_times_and_hidden_parts():
    now = datetime.fromisoformat("2026-10-31T18:00:00-07:00")
    assert parse_when("16 分钟前", now) == "2026-10-31T17:44:00-07:00"
    assert parse_when("昨天 23:02", now).startswith("2026-10-30T23:02")
    assert parse_when("2026-10-1 08:05", now).startswith("2026-10-01T08:05")
    assert parse_when("刚刚", now) is None
    post = ("Round 1 was Meeting Rooms II.\n\n您好！\n本帖隐藏的内容需要积分高于 500 才可浏览\n您当前积分为 6。\n"
            "使用VIP即刻解锁阅读权限或查看其他获取积分的方式Ken，您好！\n本帖隐藏的内容需要积分高于 500 才可浏览\n"
            "您当前积分为 6。\nVIP即刻解锁阅读权限 或 查看其他获取积分的方式\nUnlock interview details and practice with AI\n"
            "Curated Interview Questions from Top Companies\n\nGot rejected after three days.")
    text, partial = visible_text(post)
    assert partial and text.count(HIDDEN_MARK) == 1 and "积分高于" not in text and "VIP" not in text
    assert text.startswith("Round 1 was Meeting Rooms II.") and text.endswith("Got rejected after three days.")
    assert visible_text("All visible.") == ("All visible.", False)


def test_onepoint3acres_skips_without_a_profile_or_when_the_profile_is_open(tmp_path, monkeypatch):
    con = OnePoint3Acres("onepoint3acres", {}, _http({}))
    assert con.list_updates({}).skipped.startswith("no [onepoint3acres]")
    profile = tmp_path / "profile"
    profile.mkdir()
    os.symlink("otherhost-1234", profile / "SingletonLock")
    monkeypatch.setenv("SDE_SETTING__ONEPOINT3ACRES__CHROME_PROFILE", str(profile))
    from sde_rag import settings
    settings.reset_caches()
    con = OnePoint3Acres("onepoint3acres", {}, _http({}))
    assert "in use" in con.list_updates({}).skipped


def test_a_skipped_listing_is_neither_success_nor_failure():
    class Skipping(FakeConnector):
        def list_updates(self, state):
            return ListResult([], state, skipped="profile in use")

    st = sync_source("s", Skipping("s", {}))
    assert st["skipped"] == "profile in use"
    row = db.conn().execute("SELECT last_success_at, consecutive_failures FROM sources WHERE name = 's'").fetchone()
    assert row["last_success_at"] is None and row["consecutive_failures"] == 0
    assert source_due("s", 12)                               # tried again next run, without backoff


def test_download_cap_leaves_the_rest_for_the_next_run():
    docs = {f"d{i}": {"version": "1", "content": PREP_MD + f"\n## Q{i}?\nA.\n"} for i in range(5)}
    con = FakeConnector("s", docs)
    con.cfg = {"max_downloads_per_run": 2}
    st = sync_source("s", con)
    assert st["downloaded"] == 2 and st["deferred"] == 3 and len(con.finalized_failed) == 3
    st = sync_source("s", con)
    assert st["downloaded"] == 2 and st["seen"] == 2


def test_run_hours_keep_a_source_to_its_run(monkeypatch):
    other = (datetime.now().astimezone().hour + 12) % 24
    monkeypatch.setattr(ingest_graph, "source_configs", lambda: {
        "evening": {"type": "fake", "run_hours": [other]}, "any": {"type": "fake"}})
    sends = ingest_graph.plan({"run_id": 1, "owner": "o"})
    assert [s.arg["source"] for s in sends] == ["any"]
    forced = ingest_graph.plan({"run_id": 1, "owner": "o", "force": True})
    assert sorted(s.arg["source"] for s in forced) == ["any", "evening"]


def test_feed_fragments_become_markdown():
    md = fragment_to_markdown("<h2>Round 2</h2><p>Design <b>Twitter</b>&nbsp;feed.</p><ul><li>fan-out</li><li>cache</li></ul>")
    assert md == "## Round 2\n\nDesign **Twitter** feed.\n\n- fan-out\n- cache"
    assert parse_feed(MEDIUM_TAG)[0].published == "2026-10-29T07:36:22+00:00"
