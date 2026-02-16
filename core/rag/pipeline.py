from typing import List, Tuple, Dict, Any, Optional
import time
from core.rag.interface import RAGInterface
from core.rag.context import RAGDocument, RAGContextBuilder
from core.rag.retriever import RAGRetriever
from core.rag.components import QueryRewriter, QueryExpander, DomainRouter, ScoreReranker, DocFilter
from core.rag.repair import AnswerCritic, RepairStrategy
from core.trace import current_step_trace, RAGQueryTrace

class RAGPipeline(RAGInterface):
    def __init__(
        self,
        retriever: RAGRetriever,
        rewriter: QueryRewriter,
        expander: QueryExpander,
        router: DomainRouter,
        reranker: ScoreReranker,
        filterer: DocFilter,
        context_builder: RAGContextBuilder,
        max_docs: int = 10,
        max_tokens: int = 4000,
        critic: Optional[AnswerCritic] = None,
        repair_strategy: Optional[RepairStrategy] = None,
        max_retries: int = 2,
        min_score: float = 0.6,
        rerank_threshold: float = 0.0
    ):
        self.retriever = retriever
        self.rewriter = rewriter
        self.expander = expander
        self.router = router
        self.reranker = reranker
        self.filterer = filterer
        self.context_builder = context_builder
        self.max_docs = max_docs
        self.max_tokens = max_tokens
        
        # Repair Loop
        self.critic = critic
        self.repair_strategy = repair_strategy
        self.max_retries = max_retries

        # Policy Settings
        self.min_score = min_score
        self.rerank_threshold = rerank_threshold

    async def run(self, query: str, domain: str = "default") -> Tuple[str, List[RAGDocument]]:
        start_time = time.time()
        
        # Repair Loop State
        current_query = query
        attempt = 1
        
        rewritten_query = None
        used_domains = []
        found_docs_count = 0
        final_context = ""
        final_docs = []
        
        critique_passed = None
        strategy_used = None
        
        while attempt <= (self.max_retries + 1):
            try:
                # 1. Rewrite
                rewritten_query = await self.rewriter.rewrite(current_query)
                
                # 2. Expand
                queries = await self.expander.expand(rewritten_query)
                
                # 3. Route
                routed = self.router.route(rewritten_query)
                if domain != "default" and domain not in routed:
                     routed.append(domain)
                used_domains = routed
                
                # 4. Multi-Retrieve
                all_docs = []
                for q in queries:
                    for d in used_domains:
                        docs = await self.retriever.retrieve(q, collection=d)
                        all_docs.extend(docs)
                        
                found_docs_count = len(all_docs)

                # Deduplicate
                seen = set()
                unique_docs = []
                for d in all_docs:
                    h = hash(d.content)
                    if h not in seen:
                        seen.add(h)
                        unique_docs.append(d)
                
                # 5. Rerank (with Policy Threshold)
                reranked = self.reranker.rerank(unique_docs, threshold=self.rerank_threshold)
                
                # 6. Filter (with Policy Threshold)
                filtered = self.filterer.filter(reranked, min_score=self.min_score)
                
                # 7. Compress/Format
                compressed = self.context_builder.compress(filtered, max_tokens=self.max_tokens)
                formatted_context = self.context_builder.format(compressed)
                
                # 8. CRITIQUE (Repair Loop)
                if self.critic and self.repair_strategy and attempt <= self.max_retries:
                    is_good = await self.critic.critique(query, formatted_context)
                    if is_good:
                        critique_passed = True
                        final_context = formatted_context
                        final_docs = compressed
                        break # Good enough
                    else:
                         critique_passed = False
                         # Apply Repair
                         current_query = self.repair_strategy.suggest_fix(query, attempt)
                         strategy_used = current_query
                         attempt += 1
                         continue
                
                # Default success or max retries reached
                final_context = formatted_context
                final_docs = compressed
                break
                
            except Exception as e:
                # If error, maybe break or log?
                # For now, let's break to avoid infinite loops on error
                raise e

        # Trace Recording
        try:
            trace = current_step_trace.get()
            if trace:
                trace.rag_queries.append(RAGQueryTrace(
                    query=query,
                    rewritten_query=rewritten_query,
                    domains=used_domains,
                    latency_ms=(time.time() - start_time) * 1000,
                    found_docs=found_docs_count,
                    used_docs=len(final_docs),
                    sources=[d.source for d in final_docs],
                    attempt=attempt,
                    critique_passed=critique_passed,
                    repair_strategy=strategy_used
                ))
        except Exception:
            pass

        return final_context, final_docs

    async def search(self, collection_name: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Legacy support for skills using direct search.
        """
        start_time = time.time()
        docs = []
        try:
            # We can route this through the retriever directly
            docs = await self.retriever.retrieve(query, collection=collection_name, limit=limit)
            
            # Convert to dict for legacy compatibility
            return [
                {
                    "content": d.content,
                    "score": d.score,
                    "metadata": d.metadata
                }
                for d in docs
            ]
        finally:
             try:
                trace = current_step_trace.get()
                if trace:
                    trace.rag_queries.append(RAGQueryTrace(
                        query=query,
                        rewritten_query=None,
                        domains=[collection_name],
                        latency_ms=(time.time() - start_time) * 1000,
                        found_docs=len(docs),
                        used_docs=len(docs),
                        sources=[d.source for d in docs]
                    ))
             except Exception:
                pass
