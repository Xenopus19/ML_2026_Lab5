import asyncio
from mcp.server.fastmcp import FastMCP
from sqlmodel import select
from database import get_session, Book, Author

# Initialize the MCP Server
mcp = FastMCP("Library MCP Server")


# ---------------------------------------------------------------------------
# Tool 1: Read Data (Search)
# ---------------------------------------------------------------------------
@mcp.tool()
async def search_books(title_query: str) -> str:
    """
    Search for books in the library database by their title.
    Returns a formatted list of matching books and their publication years.
    """
    async for session in get_session():
        # Using SQLModel to search where the title contains the query string
        statement = select(Book).where(Book.title.contains(title_query))
        result = await session.execute(statement)
        books = result.scalars().all()

        if not books:
            return f"No books found matching '{title_query}'."

        # Format the output cleanly so the LLM can read it easily
        response = f"Found {len(books)} books:\n"
        for book in books:
            year = book.published_year or "Unknown Year"
            response += f"- ID: {book.book_id} | {book.title} ({year})\n"

        return response


# ---------------------------------------------------------------------------
# Tool 2: Write Data (Insert)
# ---------------------------------------------------------------------------
@mcp.tool()
async def add_new_author(name: str, nationality: str = None) -> str:
    """
    Adds a new author to the library database.
    Always use this tool before adding a book if the author does not exist yet.
    """
    async for session in get_session():
        new_author = Author(name=name, nationality=nationality)
        session.add(new_author)
        await session.commit()
        await session.refresh(new_author)  # Refreshes to get the new author_id

        return f"Successfully added author '{new_author.name}' with ID {new_author.author_id}."

@mcp.tool()
async def add_new_book(title: str, published_year: int = None) -> str:
    """
    Adds a new book to the library database.
    Always use this tool after ensuring the author exists.
    """
    async for session in get_session():
        new_book = Book( title=title, published_year=published_year)
        session.add(new_book)
        await session.commit()
        await session.refresh(new_book)  # Refreshes to get the new book_id

        return f"Successfully added book '{new_book.title}' with ID {new_book.book_id}."

@mcp.tool()
async def link_book_to_author(book_id: int, author_id: int) -> str:
    """
    Links an existing book to an existing author in the library database.
    Always use this tool after both the book and author exist.
    """
    async for session in get_session():
        book = await session.get(Book, book_id)
        author = await session.get(Author, author_id)

        if not book:
            return f"No book found with ID {book_id}."
        if not author:
            return f"No author found with ID {author_id}."

        await session.refresh(book, ["authors"])
        book.authors.append(author)
        await session.commit()

        return f"Successfully linked book '{book.title}' (ID {book.book_id}) to author '{author.name}' (ID {author.author_id})."


# ---------------------------------------------------------------------------
# Run the Server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # FastMCP uses standard input/output (stdio) by default to communicate
    # with the MCP client (like Claude Desktop).
    mcp.run()
