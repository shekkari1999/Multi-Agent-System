"""Music domain tool definitions."""

from __future__ import annotations

from typing import List

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool, tool


def _escape_sql_like(value: str) -> str:
    return value.replace("'", "''")


def create_music_tools(db: SQLDatabase) -> List[BaseTool]:
    """Create all music catalog tools bound to the provided database."""

    @tool
    def get_albums_by_artist(artist: str) -> str:
        """Get albums by an artist."""
        artist = _escape_sql_like(artist)
        return db.run(
            f"""
            SELECT Album.Title, Artist.Name
            FROM Album
            JOIN Artist ON Album.ArtistId = Artist.ArtistId
            WHERE Artist.Name LIKE '%{artist}%';
            """,
            include_columns=True,
        )

    @tool
    def get_tracks_by_artist(artist: str) -> str:
        """Get songs by an artist (or similar artists)."""
        artist = _escape_sql_like(artist)
        return db.run(
            f"""
            SELECT Track.Name as SongName, Artist.Name as ArtistName
            FROM Album
            LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
            LEFT JOIN Track ON Track.AlbumId = Album.AlbumId
            WHERE Artist.Name LIKE '%{artist}%';
            """,
            include_columns=True,
        )

    @tool
    def get_songs_by_genre(genre: str) -> str:
        """Fetch songs that match a specific genre."""
        genre = _escape_sql_like(genre)
        return db.run(
            f"""
            SELECT Track.Name as SongName, Artist.Name as ArtistName
            FROM Track
            JOIN Genre ON Track.GenreId = Genre.GenreId
            LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
            LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
            WHERE Genre.Name LIKE '%{genre}%'
            GROUP BY Track.Name, Artist.Name
            LIMIT 8;
            """,
            include_columns=True,
        )

    @tool
    def check_for_songs(song_title: str) -> str:
        """Check whether a song exists by name."""
        song_title = _escape_sql_like(song_title)
        return db.run(
            f"""
            SELECT *
            FROM Track
            WHERE Name LIKE '%{song_title}%';
            """,
            include_columns=True,
        )

    return [
        get_albums_by_artist,
        get_tracks_by_artist,
        get_songs_by_genre,
        check_for_songs,
    ]
