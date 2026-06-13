"""
Shared FastAPI dependencies for dependency injection.
Provides DBService instances to route handlers.
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from backend.app.database.connection import get_db
from backend.app.services.db_service import DBService


def get_db_service(db: Session = Depends(get_db)) -> DBService:
    """Dependency that provides a DBService instance."""
    return DBService(db)
