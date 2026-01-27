import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import yaml

from src.infrastructure.vector_store import VectorStore
from src.utils.llm_client import LLMClient
from src.utils.config import load_config


@dataclass
class TextChunk:
    id: str
    content: str
    source: str
    author: Optional[str] = None
    work: Optional[str] = None
    chapter: Optional[str] = None


@dataclass
class TaggedChunk:
    chunk: TextChunk
    classical_tags: List[str]
    modern_tags: List[str]
    themes: List[str]


class IngestionPipeline:
    def __init__(
        self,
        vectors: VectorStore,
        llm: Optional[LLMClient] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.vectors = vectors
        self.llm = llm or LLMClient()
        self.config = config or load_config()
        self.chunk_size = self.config.get("rag", {}).get("chunk_size", 500)
        self.chunk_overlap = self.config.get("rag", {}).get("chunk_overlap", 50)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompts_path = Path("config/prompts.yaml")
        if prompts_path.exists():
            with open(prompts_path) as f:
                return yaml.safe_load(f)
        return {}

    def ingest_stoic_text(
        self,
        file_path: str,
        author: str,
        work: str,
        tag_with_llm: bool = True
    ) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        chunks = self._chunk_text(content, source=str(path), author=author, work=work)

        if tag_with_llm:
            tagged_chunks = [self._tag_chunk(chunk) for chunk in chunks]
        else:
            tagged_chunks = [
                TaggedChunk(chunk=c, classical_tags=[], modern_tags=[], themes=[])
                for c in chunks
            ]

        return self._store_chunks(tagged_chunks, collection="stoic_wisdom")

    def ingest_psychoanalysis_text(
        self,
        file_path: str,
        author: str,
        work: str,
        tag_with_llm: bool = True
    ) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        chunks = self._chunk_text(content, source=str(path), author=author, work=work)

        if tag_with_llm:
            tagged_chunks = [self._tag_chunk(chunk, is_psych=True) for chunk in chunks]
        else:
            tagged_chunks = [
                TaggedChunk(chunk=c, classical_tags=[], modern_tags=[], themes=[])
                for c in chunks
            ]

        return self._store_chunks(tagged_chunks, collection="psychoanalysis")

    def ingest_directory(
        self,
        directory: str,
        collection: str,
        author: str,
        work: str,
        extensions: List[str] = [".txt", ".md"],
        tag_with_llm: bool = True
    ) -> int:
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        total = 0
        for ext in extensions:
            for file_path in path.glob(f"**/*{ext}"):
                if collection == "stoic_wisdom":
                    total += self.ingest_stoic_text(
                        str(file_path), author, work, tag_with_llm
                    )
                else:
                    total += self.ingest_psychoanalysis_text(
                        str(file_path), author, work, tag_with_llm
                    )
        return total

    def _chunk_text(
        self,
        text: str,
        source: str,
        author: Optional[str] = None,
        work: Optional[str] = None
    ) -> List[TextChunk]:
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append(TextChunk(
                id=str(uuid.uuid4()),
                content=chunk_text,
                source=source,
                author=author,
                work=work
            ))
            
            i += self.chunk_size - self.chunk_overlap

        return chunks

    def _tag_chunk(self, chunk: TextChunk, is_psych: bool = False) -> TaggedChunk:
        prompt_template = self.prompts.get("concept_tagging", "")
        if not prompt_template:
            return TaggedChunk(
                chunk=chunk,
                classical_tags=[],
                modern_tags=[],
                themes=[]
            )

        prompt = prompt_template.format(passage=chunk.content)

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are tagging philosophical passages for retrieval.",
                model=self.config["models"]["main"],
                temperature=0.3,
                max_tokens=300,
                json_mode=True
            )

            data = json.loads(response)
            return TaggedChunk(
                chunk=chunk,
                classical_tags=data.get("classical_tags", []),
                modern_tags=data.get("modern_tags", []),
                themes=data.get("themes", [])
            )
        except Exception as e:
            print(f"Tagging failed for chunk {chunk.id}: {e}")
            return TaggedChunk(
                chunk=chunk,
                classical_tags=[],
                modern_tags=[],
                themes=[]
            )

    def _store_chunks(
        self,
        tagged_chunks: List[TaggedChunk],
        collection: str
    ) -> int:
        if not tagged_chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for tc in tagged_chunks:
            ids.append(tc.chunk.id)
            documents.append(tc.chunk.content)
            
            all_tags = tc.classical_tags + tc.modern_tags + tc.themes
            metadatas.append({
                "source": tc.chunk.source,
                "author": tc.chunk.author or "",
                "work": tc.chunk.work or "",
                "classical_tags": ",".join(tc.classical_tags),
                "modern_tags": ",".join(tc.modern_tags),
                "themes": ",".join(tc.themes),
                "all_tags": ",".join(all_tags)
            })

        self.vectors.add(
            collection=collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        return len(tagged_chunks)


def ingest_stoic_highlights(
    vectors: VectorStore,
    llm: Optional[LLMClient] = None
) -> int:
    highlights = [
        {
            "author": "Marcus Aurelius",
            "work": "Meditations",
            "passages": [
                "You have power over your mind - not outside events. Realize this, and you will find strength.",
                "The happiness of your life depends upon the quality of your thoughts.",
                "Waste no more time arguing about what a good man should be. Be one.",
                "Very little is needed to make a happy life; it is all within yourself, in your way of thinking.",
                "Accept the things to which fate binds you, and love the people with whom fate brings you together, and do so with all your heart.",
                "The best revenge is to be unlike him who performed the injury.",
                "When you arise in the morning think of what a privilege it is to be alive, to think, to enjoy, to love.",
                "Never let the future disturb you. You will meet it, if you have to, with the same weapons of reason which today arm you against the present.",
                "The object of life is not to be on the side of the majority, but to escape finding oneself in the ranks of the insane.",
                "If it is not right do not do it; if it is not true do not say it.",
            ]
        },
        {
            "author": "Seneca",
            "work": "Letters from a Stoic",
            "passages": [
                "We suffer more often in imagination than in reality.",
                "It is not that we have a short time to live, but that we waste a lot of it.",
                "Luck is what happens when preparation meets opportunity.",
                "Begin at once to live, and count each separate day as a separate life.",
                "He suffers more than necessary, who suffers before it is necessary.",
                "It is not the man who has too little that is poor, but the one who hankers after more.",
                "While we are postponing, life speeds by.",
                "True happiness is to enjoy the present, without anxious dependence upon the future.",
            ]
        },
        {
            "author": "Epictetus",
            "work": "Enchiridion",
            "passages": [
                "Some things are in our control and others not. Things in our control are opinion, pursuit, desire, aversion, and our own actions. Things not in our control are body, property, reputation, command, and whatever are not our own actions.",
                "Men are disturbed not by things, but by the views which they take of them.",
                "It is difficulties that show what men are.",
                "No man is free who is not master of himself.",
                "First say to yourself what you would be; and then do what you have to do.",
                "If you want to improve, be content to be thought foolish and stupid.",
                "The key is to keep company only with people who uplift you, whose presence calls forth your best.",
            ]
        }
    ]

    pipeline = IngestionPipeline(vectors, llm)
    total = 0

    for source in highlights:
        author = source["author"]
        work = source["work"]
        
        for passage in source["passages"]:
            chunk = TextChunk(
                id=str(uuid.uuid4()),
                content=passage,
                source=f"{author} - {work}",
                author=author,
                work=work
            )
            
            tagged = pipeline._tag_chunk(chunk)
            total += pipeline._store_chunks([tagged], collection="stoic_wisdom")

    return total
