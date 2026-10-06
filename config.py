"""
Configuration module for OmniRAG Studio.
Defines hyperparameters, use case presets, API client configurations, and observability thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import os


@dataclass
class PersonaPreset:
    name: str
    description: str
    persona_prompt: str
    default_chunk_size: int
    default_overlap: int
    default_top_k: int
    default_threshold: float
    sample_documents: List[Dict[str, str]]


@dataclass
class ModelConfig:
    # Embedding Configuration
    embedding_provider: str = "tfidf"  # "tfidf", "sentence-transformers", "openai"
    sentence_model_name: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    
    # LLM Generator Configuration
    llm_provider: str = "anthropic"  # "anthropic", "openai", "offline"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    openai_model: str = "gpt-4o"
    max_tokens: int = 400
    temperature: float = 0.0
    
    # API Keys
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))


# Pre-configured Domain Datasets & Personas from PRD Section 3
STUDY_BUDDY_PRESET = PersonaPreset(
    name="Technical Exam Study Buddy",
    description="Dense academic textbooks, lecture notes, algorithms, and systems design principles.",
    persona_prompt=(
        "You are a patient, expert study partner helping a student revise for technical exams. "
        "Explain concepts rigorously yet simply, citing retrieved lecture notes using format [Source X]. "
        "Strictly ground your explanation in provided notes; if absent, state: 'I don't have that information in your revision notes.'"
    ),
    default_chunk_size=50,
    default_overlap=15,
    default_top_k=3,
    default_threshold=0.10,
    sample_documents=[
        {
            "title": "OS Lecture 04: CPU Scheduling & Multitasking",
            "content": (
                "CPU Scheduling is the process by which the operating system decides which thread or process in the ready queue "
                "is allocated CPU execution time. Preemptive scheduling allows the OS kernel to interrupt a currently running process "
                "when a higher-priority task arrives or when a time quantum expires (such as in Round Robin scheduling). "
                "In contrast, non-preemptive scheduling (e.g., First-Come-First-Served or Non-Preemptive Shortest Job First) allows "
                "the running process to hold the CPU until it voluntarily terminates or enters an I/O wait state. "
                "Starvation can occur in strict priority scheduling when low-priority jobs are perpetually superseded by high-priority arrivals. "
                "Aging is the standard mitigation technique, where the system gradually increments the priority of waiting processes over time."
            )
        },
        {
            "title": "OS Lecture 09: Virtual Memory & Page Faults",
            "content": (
                "Virtual memory maps a process's logical address space to physical RAM via page tables managed by the Memory Management Unit (MMU). "
                "When a process accesses a logical address whose page table entry indicates the page is not in physical RAM (valid-invalid bit is 0), "
                "the MMU triggers a page fault hardware trap. The OS kernel page fault handler intercepts the trap, allocates an available physical frame, "
                "reads the requested page from secondary disk storage via direct memory access, updates the page table entry, and restarts the instruction. "
                "Thrashing occurs when the system spends more time servicing page faults and swapping pages than executing instructions, "
                "typically when the aggregate working set of all processes exceeds available physical memory frames."
            )
        },
        {
            "title": "Databases Lecture 06: Concurrency Control & Two-Phase Locking",
            "content": (
                "Two-Phase Locking (2PL) is a concurrency control protocol that guarantees serializability in relational database systems. "
                "The protocol is divided into two distinct phases: the Growing Phase and the Shrinking Phase. "
                "During the Growing Phase, a transaction may acquire shared (read) locks or exclusive (write) locks, but cannot release any lock. "
                "Once the transaction releases its very first lock, it irrevocably enters the Shrinking Phase, where it may release remaining locks "
                "but cannot acquire any new locks. Strict 2PL requires that all exclusive locks be held until the transaction explicitly commits or aborts, "
                "which completely prevents cascading aborts and guarantees recoverability."
            )
        }
    ]
)

ECOMMERCE_SUPPORT_PRESET = PersonaPreset(
    name="Enterprise E-Commerce Support",
    description="Product specifications, return policies, warranty coverage, and shipping tiers.",
    persona_prompt=(
        "You are a polite, highly accurate customer support agent. "
        "Answer customer queries with strict adherence to policy rules and product specifications using format [Source X]. "
        "Never extrapolate or assume policies outside the documented terms. "
        "If uncertain or unlisted, say 'I don't have information on that policy.'"
    ),
    default_chunk_size=80,
    default_overlap=18,
    default_top_k=4,
    default_threshold=0.15,
    sample_documents=[
        {
            "title": "Global Return & Refund Policy 2026",
            "content": (
                "Customers are eligible for a full refund on eligible merchandise within 30 calendar days from the verified delivery date. "
                "All returned items must remain in original, unworn condition with manufacturer security tags attached and in original packaging. "
                "Electronics, including laptops, tablets, and headphones, have an accelerated 14-day return window and are subject to a 10% "
                "restocking fee if opened, unless determined to be dead-on-arrival by technical diagnostics. "
                "Final sale items, personalized goods, and software licenses are strictly non-refundable and exempt from returns under all circumstances."
            )
        },
        {
            "title": "Domestic & International Shipping Tiers",
            "content": (
                "Standard Shipping arrives within 3 to 5 business days and is complimentary for all domestic orders exceeding $50.00. "
                "Expedited Priority Shipping provides guaranteed 2-day delivery for a flat fee of $12.99 across all mainland postal codes. "
                "Overnight Delivery is available for orders confirmed prior to 1:00 PM Eastern Standard Time for $24.99. "
                "International orders require customs clearance which may add 2 to 7 business days beyond transit estimations; import customs duties "
                "and VAT taxes are the sole responsibility of the receiving customer unless Delivered Duty Paid (DDP) is selected at checkout."
            )
        },
        {
            "title": "Hardware Warranty & SKU Coverage Guidelines",
            "content": (
                "All flagship hardware products (SKU prefix HW-PRO) come standard with a 2-Year Limited Manufacturer Hardware Warranty covering "
                "defects in materials and workmanship. The warranty covers motherboard repairs, display panel backlight failure, and internal power supply faults. "
                "Accidental damage, including liquid spills, dropped screens, unauthorized chassis modifications, or cosmetic abrasions, is explicitly excluded. "
                "Customers enrolled in the CarePlus Protection Plan receive up to two incidents of accidental damage coverage per 12-month cycle, "
                "subject to a standardized $49.00 service deductible per claim."
            )
        }
    ]
)

PRESETS: Dict[str, PersonaPreset] = {
    "study_buddy": STUDY_BUDDY_PRESET,
    "ecommerce": ECOMMERCE_SUPPORT_PRESET
}
