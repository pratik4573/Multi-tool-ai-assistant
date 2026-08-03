"""
System prompt construction helpers for Pratik AI.

Priority:
1. Retrieved personal documents (RAG)
2. Long-term memories
3. General model knowledge
"""

from __future__ import annotations

import config


def build_system_prompt(
    retrieved_context: list[str],
    relevant_memories: list[str],
) -> str:
    """
    Build the final system prompt.

    Information priority:

    1. Personal document context (RAG)
    2. Long-term memories
    3. General knowledge

    Personal documents always override memories if they conflict.
    """

    sections = [config.SYSTEM_PROMPT.strip()]

    instructions = """
IMPORTANT RULES

1. PERSONAL DOCUMENTS ARE THE MOST TRUSTWORTHY SOURCE for questions
   ABOUT THE USER (their background, projects, preferences, history, etc.).

2. If the retrieved personal documents contain the answer to a
   question about the user, ALWAYS answer using those documents.

3. Memories are only supplementary information about the user.

4. If a memory conflicts with a retrieved document about the user,
   IGNORE THE MEMORY.

5. Never say
   "the personal context does not provide..."
   if the retrieved document clearly contains the answer.

6. Never invent facts about the user that aren't supported by the
   retrieved documents or memories.

7. SCOPE CHECK — first decide whether the question is ABOUT THE USER
   or a GENERAL KNOWLEDGE question (people, places, facts, concepts,
   definitions, anything not about the user personally):

   - If it IS about the user, and neither the retrieved documents nor
     memories below contain the answer, politely say you don't have
     that information about the user. Do not guess.

   - If it is NOT about the user (general knowledge), the absence of
     retrieved documents or memories is expected and irrelevant —
     it does not mean you lack the answer. Answer confidently and
     directly from your own knowledge. Never refuse or say you don't
     have information for a general-knowledge question just because
     no personal document/memory was retrieved for it.

8. Only apply rules 1-6 (documents/memories as source of truth) to
   questions about the user. General knowledge questions are governed
   by rule 7's second bullet instead.
"""

    sections.append(instructions.strip())

    if retrieved_context:

        context = "\n\n".join(
            f"DOCUMENT {i + 1}:\n{chunk}"
            for i, chunk in enumerate(retrieved_context)
        )

        sections.append(
            "RETRIEVED PERSONAL DOCUMENTS:\n"
            + context
        )

    if relevant_memories:

        memory = "\n".join(
            f"- {m}"
            for m in relevant_memories
        )

        sections.append(
            "LONG-TERM MEMORIES (secondary source):\n"
            + memory
        )

    return "\n\n".join(sections)