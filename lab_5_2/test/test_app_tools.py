import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, ToolMessage
from streamlit.testing.v1 import AppTest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import server


class AppToolTests(unittest.TestCase):
    def test_book_creation_and_linking_in_separate_rounds(self):
        class FakeModel:
            def __init__(self, **kwargs):
                self.calls = []
                self.tool_names = []

            def bind_tools(self, tools):
                self.tool_names = [tool.name for tool in tools]
                return self

            def invoke(self, messages):
                self.calls.append(list(messages))
                if len(self.calls) == 1:
                    return AIMessage(content="", tool_calls=[{
                        "name": "tool_add_new_book",
                        "args": {"title": "New book"},
                        "id": "book-call",
                    }])
                if len(self.calls) == 2:
                    return AIMessage(content="", tool_calls=[{
                        "name": "tool_link_book_to_author",
                        "args": {"book_id": 42, "author_id": 7},
                        "id": "link-call",
                    }])
                return AIMessage(content="Book linked.")

        model = FakeModel()
        with (
            patch("langchain_ollama.ChatOllama", return_value=model),
            patch.object(server, "add_new_book", new_callable=AsyncMock) as add_book,
            patch.object(server, "link_book_to_author", new_callable=AsyncMock) as link_book,
        ):
            add_book.return_value = "Successfully added book 'New book' with ID 42."
            link_book.return_value = "Successfully linked book to author."
            app = AppTest.from_file(str(SRC / "app.py")).run()
            app.chat_input[0].set_value("Add New book and link it to author 7").run()

        self.assertFalse(app.exception)
        self.assertEqual(model.tool_names, [
            "tool_search_books", "tool_add_new_author",
            "tool_add_new_book", "tool_link_book_to_author",
        ])
        add_book.assert_awaited_once_with("New book", None)
        link_book.assert_awaited_once_with(42, 7)
        self.assertEqual(model.calls[1][-1].content, add_book.return_value)
        self.assertIsInstance(model.calls[2][-1], ToolMessage)
        self.assertEqual(model.calls[2][-1].content, link_book.return_value)
        self.assertEqual(app.session_state["messages"][-1]["content"], "Book linked.")


class LinkBookTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_authors_before_linking(self):
        class Book:
            book_id = 42
            title = "New book"

            def __init__(self):
                self.loaded = False
                self._authors = []

            @property
            def authors(self):
                if not self.loaded:
                    raise AssertionError("authors must be loaded before access")
                return self._authors

        book = Book()
        author = server.Author(author_id=7, name="Author")
        session = AsyncMock()
        session.get.side_effect = [book, author]

        async def refresh(instance, attributes):
            self.assertIs(instance, book)
            self.assertEqual(attributes, ["authors"])
            book.loaded = True

        session.refresh.side_effect = refresh

        async def get_test_session():
            yield session

        with patch.object(server, "get_session", get_test_session):
            result = await server.link_book_to_author(42, 7)

        self.assertEqual(book.authors, [author])
        session.commit.assert_awaited_once()
        self.assertIn("Successfully linked", result)


if __name__ == "__main__":
    unittest.main()
