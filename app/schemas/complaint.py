"""
schemas/complaint.py — the canonical domain models for a pharmaceutical complaint.

Why this file is the source of truth:
  Every layer (LangGraph extraction prompt, database ORM, API response) derives
  its shape from these models.  Keeping them in one place means adding a field
  later is a single-file change; it also gives interviewers a clean place to
  ask "walk me through the domain".

Design decisions worth remembering:
  - Enums are Python str-enums so FastAPI serialises them as plain strings in
    JSON (not {"value": "Email"}).
  - All extracted fields are Optional with None defaults because the AI may not
    find every value in a given complaint — the completeness_check node handles
    the missing-field logic.
  - ComplaintExtraction is the Pydantic model we pass to ChatGroq's
    structured-output / function-calling API; it must be JSON-serialisable and
    must match the form fields spec exactly.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class ComplaintSource(str, Enum):
    """Channel through which the complaint arrived."""
    EMAIL = "Email"
    PHONE = "Phone"
    PORTAL = "Portal"
    DISTRIBUTOR = "Distributor"
    REGULATORY = "Regulatory"


class ComplaintType(str, Enum):
    """High-level category of the complaint — used for CAPA routing."""
    QUALITY_DEFECT = "Quality Defect"
    PACKAGING = "Packaging"
    LABELING = "Labeling"
    ADVERSE_EVENT = "Adverse Event"
    DELIVERY_DOCUMENTATION = "Delivery/Documentation"


class Severity(str, Enum):
    """
    Pharma severity levels aligned with ICH Q10 / GMP terminology.
    Critical = patient safety / regulatory risk.
    Major    = significant quality impact but no immediate patient risk.
    Minor    = cosmetic / administrative issues.
    """
    CRITICAL = "Critical"
    MAJOR = "Major"
    MINOR = "Minor"


class Priority(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


# ── Core extracted payload ────────────────────────────────────────────────────

class ComplaintExtraction(BaseModel):
    """
    The structured data the LLM must extract from the raw complaint text.

    This model is passed directly to ChatGroq's with_structured_output() so
    every field name, type, and description becomes part of the JSON schema
    sent to the model.  The Field(description=...) strings are the hints the
    LLM reads to know what to look for — they're not just for humans.

    All fields are Optional so the model can return null for genuinely missing
    information rather than hallucinating a value.
    """

    # Origin & Customer
    complaint_source: Optional[ComplaintSource] = Field(
        None,
        description="Channel through which the complaint was received "
                    "(Email, Phone, Portal, Distributor, Regulatory).",
    )
    customer_name: Optional[str] = Field(
        None, description="Full name of the customer or organisation raising the complaint."
    )

    # Product & Batch
    product_name: Optional[str] = Field(
        None, description="Commercial or INN name of the pharmaceutical product."
    )
    product_strength_grade: Optional[str] = Field(
        None,
        description="Strength (e.g. '500 mg') or API grade (e.g. 'USP') of the product.",
    )
    batch_lot_number: Optional[str] = Field(
        None, description="Manufacturer's batch or lot number on the label."
    )
    manufacturing_date: Optional[date] = Field(
        None, description="Date of manufacture as printed on the product/label (YYYY-MM-DD)."
    )
    expiry_date: Optional[date] = Field(
        None, description="Expiry / best-before date of the batch (YYYY-MM-DD)."
    )
    quantity_affected: Optional[str] = Field(
        None,
        description="Quantity involved in the complaint including unit, e.g. '50 kg' or '200 tablets'.",
    )

    # Complaint Details
    complaint_type: Optional[ComplaintType] = Field(
        None,
        description="Category of the complaint: Quality Defect, Packaging, Labeling, "
                    "Adverse Event, or Delivery/Documentation.",
    )
    complaint_date: Optional[date] = Field(
        None, description="Date the complaint was raised or received (YYYY-MM-DD)."
    )
    detailed_complaint_description: Optional[str] = Field(
        None,
        description="Full verbatim or paraphrased description of the issue as reported.",
    )

    # Initial Assessment (LLM fills these in risk_classification node)
    initial_severity: Optional[Severity] = Field(
        None, description="Initial severity assessment: Critical, Major, or Minor."
    )
    priority: Optional[Priority] = Field(
        None, description="Handling priority: High, Medium, or Low."
    )

    @field_validator("complaint_source", mode="before")
    @classmethod
    def _coerce_source(cls, v):
        if not v:
            return None
        if isinstance(v, ComplaintSource):
            return v
        s = str(v).strip().lower()
        if "email" in s:
            return ComplaintSource.EMAIL
        if "phone" in s or "call" in s:
            return ComplaintSource.PHONE
        if "portal" in s or "web" in s or "online" in s:
            return ComplaintSource.PORTAL
        if "distrib" in s or "vendor" in s or "wholesaler" in s:
            return ComplaintSource.DISTRIBUTOR
        if "regulat" in s or "fda" in s or "authority" in s or "health" in s:
            return ComplaintSource.REGULATORY
        for cs in ComplaintSource:
            if cs.value.lower() == s:
                return cs
        return None

    @field_validator("complaint_type", mode="before")
    @classmethod
    def _coerce_type(cls, v):
        if not v:
            return None
        if isinstance(v, ComplaintType):
            return v
        s = str(v).strip().lower()
        if "defect" in s or "quality" in s or "contam" in s or "color" in s or "discolor" in s:
            return ComplaintType.QUALITY_DEFECT
        if "pack" in s or "seal" in s or "bottle" in s or "strip" in s or "blister" in s:
            return ComplaintType.PACKAGING
        if "label" in s or "print" in s or "leaflet" in s or "artwork" in s:
            return ComplaintType.LABELING
        if "adverse" in s or "side effect" in s or "reaction" in s:
            return ComplaintType.ADVERSE_EVENT
        if "delivery" in s or "document" in s or "ship" in s or "invoice" in s:
            return ComplaintType.DELIVERY_DOCUMENTATION
        for ct in ComplaintType:
            if ct.value.lower() == s:
                return ct
        return None

    @field_validator("initial_severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v):
        if not v:
            return None
        if isinstance(v, Severity):
            return v
        s = str(v).strip().lower()
        if "crit" in s:
            return Severity.CRITICAL
        if "maj" in s:
            return Severity.MAJOR
        if "min" in s:
            return Severity.MINOR
        for sv in Severity:
            if sv.value.lower() == s:
                return sv
        return None

    @field_validator("priority", mode="before")
    @classmethod
    def _coerce_priority(cls, v):
        if not v:
            return None
        if isinstance(v, Priority):
            return v
        s = str(v).strip().lower()
        if "high" in s:
            return Priority.HIGH
        if "med" in s:
            return Priority.MEDIUM
        if "low" in s:
            return Priority.LOW
        for p in Priority:
            if p.value.lower() == s:
                return p
        return None


# ── AI-generated analysis fields ──────────────────────────────────────────────

class ComplaintAnalysis(BaseModel):
    """
    Fields produced by the later graph nodes (risk, CAPA, summary).
    Kept separate from ComplaintExtraction so the API response can clearly
    distinguish 'what we extracted' from 'what the AI concluded'.
    """
    risk_justification: Optional[str] = Field(
        None,
        description="Short justification for the assigned severity/priority rating.",
    )
    duplicate_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs of existing complaints with high cosine similarity.",
    )
    duplicate_scores: list[float] = Field(
        default_factory=list,
        description="Similarity scores (0-1) corresponding to duplicate_ids.",
    )
    root_cause_hypothesis: Optional[str] = Field(
        None, description="AI-suggested probable root cause of the complaint."
    )
    capa_recommendation: Optional[str] = Field(
        None,
        description="Corrective and Preventive Action recommended by the AI.",
    )
    summary: Optional[str] = Field(
        None,
        description="2-3 sentence human-readable summary for the chat sidebar.",
    )


# ── Full complaint record ─────────────────────────────────────────────────────

class Complaint(BaseModel):
    """
    The persisted record stored in Postgres after the user confirms the form.
    Combines extraction + analysis + system metadata.
    This mirrors the DB table columns (the ORM model in Phase 4 will match it).
    """
    id: UUID = Field(default_factory=uuid4)
    extraction: ComplaintExtraction = Field(default_factory=ComplaintExtraction)
    analysis: ComplaintAnalysis = Field(default_factory=ComplaintAnalysis)
    raw_text: Optional[str] = Field(
        None, description="Original complaint text kept for audit purposes."
    )
    is_confirmed: bool = Field(
        False,
        description="True once the user submits the confirmed form — "
                    "only confirmed complaints are persisted.",
    )
