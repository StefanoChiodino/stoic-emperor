import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import create_engine, String, Integer, Float, DateTime, Text, ForeignKey, JSON, Index, select, func, desc
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session as SASession

from src.models.schemas import User, Session, Message, SemanticInsight, PsychUpdate, CondensedSummary


class Base(DeclarativeBase):
    pass


class UserModel(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class SessionModel(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default=dict)

    __table_args__ = (Index("idx_sessions_user", "user_id"),)


class MessageModel(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"))
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    psych_update: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    semantic_processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_messages_session", "session_id"),
        Index("idx_messages_semantic", "semantic_processed_at"),
    )


class ProfileModel(Base):
    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    version: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    consensus_log: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class SemanticInsightModel(Base):
    __tablename__ = "semantic_insights"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    source_message_id: Mapped[str] = mapped_column(String, ForeignKey("messages.id"))
    assertion: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    __table_args__ = (Index("idx_insights_user", "user_id"),)


class CondensedSummaryModel(Base):
    __tablename__ = "condensed_summaries"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"))
    level: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    period_start: Mapped[datetime] = mapped_column(DateTime)
    period_end: Mapped[datetime] = mapped_column(DateTime)
    source_message_count: Mapped[int] = mapped_column(Integer)
    source_word_count: Mapped[int] = mapped_column(Integer)
    source_summary_ids: Mapped[Optional[list]] = mapped_column(JSON, default=list)
    consensus_log: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_summaries_user_level", "user_id", "level"),
        Index("idx_summaries_period", "user_id", "period_end"),
    )


class SchemaVersionModel(Base):
    __tablename__ = "schema_version"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Database:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///./data/stoic_emperor.db")

        if not self.database_url.startswith(("sqlite", "postgresql", "postgres")):
            self.database_url = f"sqlite:///{self.database_url}"

        if self.database_url.startswith("sqlite"):
            from pathlib import Path
            db_path = self.database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        is_sqlite = self.database_url.startswith("sqlite")
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            **({"pool_size": 5, "max_overflow": 10} if not is_sqlite else {}),
        )
        self._session_factory = sessionmaker(bind=self.engine)

        Base.metadata.create_all(self.engine)
        self._ensure_schema_version()

    def _ensure_schema_version(self) -> None:
        with self._session() as session:
            existing = session.get(SchemaVersionModel, 4)
            if not existing:
                session.merge(SchemaVersionModel(version=4))

    @contextmanager
    def _session(self):
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_user(self, user: User) -> User:
        with self._session() as session:
            model = UserModel(id=user.id, created_at=user.created_at)
            session.add(model)
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        with self._session() as session:
            model = session.get(UserModel, user_id)
            if model:
                return User(id=model.id, created_at=model.created_at)
        return None

    def get_or_create_user(self, user_id: str) -> User:
        user = self.get_user(user_id)
        if not user:
            user = User(id=user_id)
            self.create_user(user)
        return user

    def create_session(self, session_obj: Session) -> Session:
        with self._session() as session:
            model = SessionModel(
                id=session_obj.id,
                user_id=session_obj.user_id,
                created_at=session_obj.created_at,
                metadata_=session_obj.metadata,
            )
            session.add(model)
        return session_obj

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._session() as session:
            model = session.get(SessionModel, session_id)
            if model:
                return Session(
                    id=model.id,
                    user_id=model.user_id,
                    created_at=model.created_at,
                    metadata=model.metadata_ or {},
                )
        return None

    def get_latest_session(self, user_id: str) -> Optional[Session]:
        with self._session() as session:
            stmt = (
                select(SessionModel)
                .where(SessionModel.user_id == user_id)
                .order_by(desc(SessionModel.created_at))
                .limit(1)
            )
            model = session.scalars(stmt).first()
            if model:
                return Session(
                    id=model.id,
                    user_id=model.user_id,
                    created_at=model.created_at,
                    metadata=model.metadata_ or {},
                )
        return None

    def save_message(self, message: Message) -> Message:
        with self._session() as session:
            psych_dict = message.psych_update.model_dump() if message.psych_update else None
            model = MessageModel(
                id=message.id,
                session_id=message.session_id,
                role=message.role,
                content=message.content,
                psych_update=psych_dict,
                created_at=message.created_at,
                semantic_processed_at=message.semantic_processed_at,
            )
            session.add(model)
        return message

    def get_session_messages(self, session_id: str) -> List[Message]:
        with self._session() as session:
            stmt = (
                select(MessageModel)
                .where(MessageModel.session_id == session_id)
                .order_by(MessageModel.created_at)
            )
            models = session.scalars(stmt).all()
            return [self._model_to_message(m) for m in models]

    def get_unprocessed_messages(self, user_id: str) -> List[Message]:
        with self._session() as session:
            stmt = (
                select(MessageModel)
                .join(SessionModel, MessageModel.session_id == SessionModel.id)
                .where(SessionModel.user_id == user_id)
                .where(MessageModel.semantic_processed_at.is_(None))
                .where(MessageModel.psych_update.isnot(None))
                .order_by(MessageModel.created_at)
            )
            models = session.scalars(stmt).all()
            return [self._model_to_message(m) for m in models]

    def mark_message_processed(self, message_id: str) -> None:
        with self._session() as session:
            model = session.get(MessageModel, message_id)
            if model:
                model.semantic_processed_at = datetime.now()

    def save_semantic_insight(self, insight: SemanticInsight) -> SemanticInsight:
        with self._session() as session:
            model = SemanticInsightModel(
                id=insight.id,
                user_id=insight.user_id,
                source_message_id=insight.source_message_id,
                assertion=insight.assertion,
                confidence=insight.confidence,
                created_at=insight.created_at,
            )
            session.add(model)
        return insight

    def get_user_insights(self, user_id: str) -> List[SemanticInsight]:
        with self._session() as session:
            stmt = (
                select(SemanticInsightModel)
                .where(SemanticInsightModel.user_id == user_id)
                .order_by(desc(SemanticInsightModel.created_at))
            )
            models = session.scalars(stmt).all()
            return [
                SemanticInsight(
                    id=m.id,
                    user_id=m.user_id,
                    source_message_id=m.source_message_id,
                    assertion=m.assertion,
                    confidence=m.confidence,
                    created_at=m.created_at,
                )
                for m in models
            ]

    def count_sessions_since_last_analysis(self, user_id: str) -> int:
        with self._session() as session:
            last_profile_stmt = (
                select(func.max(ProfileModel.created_at))
                .where(ProfileModel.user_id == user_id)
            )
            last_profile_time = session.scalar(last_profile_stmt)

            if last_profile_time:
                count_stmt = (
                    select(func.count())
                    .select_from(SessionModel)
                    .where(SessionModel.user_id == user_id)
                    .where(SessionModel.created_at > last_profile_time)
                )
            else:
                count_stmt = (
                    select(func.count())
                    .select_from(SessionModel)
                    .where(SessionModel.user_id == user_id)
                )
            return session.scalar(count_stmt) or 0

    def get_latest_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._session() as session:
            stmt = (
                select(ProfileModel)
                .where(ProfileModel.user_id == user_id)
                .order_by(desc(ProfileModel.version))
                .limit(1)
            )
            model = session.scalars(stmt).first()
            if model:
                return {
                    "content": model.content,
                    "version": model.version,
                    "created_at": model.created_at,
                    "consensus_log": model.consensus_log or {},
                }
        return None

    def save_condensed_summary(self, summary: CondensedSummary) -> CondensedSummary:
        with self._session() as session:
            model = CondensedSummaryModel(
                id=summary.id,
                user_id=summary.user_id,
                level=summary.level,
                content=summary.content,
                period_start=summary.period_start,
                period_end=summary.period_end,
                source_message_count=summary.source_message_count,
                source_word_count=summary.source_word_count,
                source_summary_ids=summary.source_summary_ids,
                consensus_log=summary.consensus_log,
                created_at=summary.created_at,
            )
            session.add(model)
        return summary

    def get_condensed_summaries(self, user_id: str, level: Optional[int] = None) -> List[CondensedSummary]:
        with self._session() as session:
            stmt = select(CondensedSummaryModel).where(CondensedSummaryModel.user_id == user_id)
            if level is not None:
                stmt = stmt.where(CondensedSummaryModel.level == level).order_by(CondensedSummaryModel.period_start)
            else:
                stmt = stmt.order_by(CondensedSummaryModel.level, CondensedSummaryModel.period_start)
            models = session.scalars(stmt).all()
            return [self._model_to_condensed_summary(m) for m in models]

    def get_messages_in_range(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Message]:
        with self._session() as session:
            stmt = (
                select(MessageModel)
                .join(SessionModel, MessageModel.session_id == SessionModel.id)
                .where(SessionModel.user_id == user_id)
            )
            if start_date:
                stmt = stmt.where(MessageModel.created_at >= start_date)
            if end_date:
                stmt = stmt.where(MessageModel.created_at <= end_date)
            stmt = stmt.order_by(MessageModel.created_at)
            models = session.scalars(stmt).all()
            return [self._model_to_message(m) for m in models]

    def get_recent_messages(self, user_id: str, limit: int = 20) -> List[Message]:
        with self._session() as session:
            stmt = (
                select(MessageModel)
                .join(SessionModel, MessageModel.session_id == SessionModel.id)
                .where(SessionModel.user_id == user_id)
                .order_by(desc(MessageModel.created_at))
                .limit(limit)
            )
            models = session.scalars(stmt).all()
            return list(reversed([self._model_to_message(m) for m in models]))

    def _model_to_message(self, model: MessageModel) -> Message:
        psych = None
        if model.psych_update:
            psych = PsychUpdate(**model.psych_update)
        return Message(
            id=model.id,
            session_id=model.session_id,
            role=model.role,
            content=model.content,
            psych_update=psych,
            created_at=model.created_at,
            semantic_processed_at=model.semantic_processed_at,
        )

    def _model_to_condensed_summary(self, model: CondensedSummaryModel) -> CondensedSummary:
        return CondensedSummary(
            id=model.id,
            user_id=model.user_id,
            level=model.level,
            content=model.content,
            period_start=model.period_start,
            period_end=model.period_end,
            source_message_count=model.source_message_count,
            source_word_count=model.source_word_count,
            source_summary_ids=model.source_summary_ids or [],
            consensus_log=model.consensus_log,
            created_at=model.created_at,
        )
