"""
seed_data.py — populate the complaints table with 5 realistic pharma complaints.

Run from backend/ AFTER applying the migration:
    alembic upgrade head
    python seed_data.py

Purpose:
  1. Gives the duplicate_detection node something to search against from day 1.
  2. Provides demo data for the frontend walkthrough.
  3. Includes deliberate near-duplicates (complaints 1 and 2 describe the same
     Amoxicillin seal defect from different customers) so the duplicate-detection
     node actually fires during testing.

Why asyncio.run() here (not a sync script)?
  Our engine is async-only (asyncpg driver).  We wrap everything in an async
  main() and call asyncio.run() — the standard pattern for running async code
  in a script entrypoint.

Design note on embeddings:
  We embed the detailed_complaint_description of each complaint so the
  pgvector cosine-similarity search has real vectors to compare against.
  Seeding without embeddings would mean duplicate_detection always returns
  empty results, making it untestable without real uploaded complaints.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

# ── Make app importable when running from backend/ ────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from sqlalchemy import text
from app.db.engine import get_async_session_factory
from app.db.models import ComplaintORM
from app.graph.embedder import embed_text        # Phase 4 embedder


# ── Seed records ──────────────────────────────────────────────────────────────
# Each dict maps exactly to ComplaintORM columns.
# complaints 0 and 1 are near-duplicates (same product/batch, different reporters)
# to verify that duplicate_detection fires above the 0.85 threshold.

SEED_COMPLAINTS: list[dict] = [
    {
        "product_name": "Amoxicillin Trihydrate Capsules",
        "batch_lot_number": "BX-20240315-002",
        "customer_name": "MedSupply Europe GmbH",
        "complaint_date": "2024-03-15",
        "initial_severity": "Major",
        "priority": "High",
        "extraction_data": {
            "complaint_source": "Email",
            "customer_name": "MedSupply Europe GmbH",
            "product_name": "Amoxicillin Trihydrate Capsules",
            "product_strength_grade": "500 mg / USP",
            "batch_lot_number": "BX-20240315-002",
            "manufacturing_date": "2024-01-15",
            "expiry_date": "2026-01-14",
            "quantity_affected": "50 units",
            "complaint_type": "Packaging",
            "complaint_date": "2024-03-15",
            "detailed_complaint_description": (
                "Approximately 50 capsules out of 1,000 units had broken induction seals "
                "on HDPE bottle closures, identified during incoming inspection at our "
                "Frankfurt warehouse. No patient exposure reported."
            ),
            "initial_severity": "Major",
            "priority": "High",
        },
        "risk_justification": (
            "Packaging seal failure on a distributed batch is Major severity: product "
            "integrity compromised but no direct patient adverse event reported. High "
            "priority because the batch is still in active distribution."
        ),
        "root_cause_hypothesis": (
            "Probable cause: sealing machine temperature out of specification during "
            "packaging run for batch BX-20240315-002, resulting in incomplete induction seals."
        ),
        "capa_text": (
            "Corrective: Quarantine batch BX-20240315-002, conduct 100% visual inspection. "
            "Preventive: Implement in-process sealing integrity checks every 30 min; "
            "calibrate sealing equipment weekly; add seal integrity to batch release checklist."
        ),
        "summary_text": (
            "A Packaging complaint was received from MedSupply Europe GmbH regarding "
            "Amoxicillin 500mg (Batch BX-20240315-002) with 50 units showing broken seals. "
            "Severity: Major / Priority: High. Quarantine and sealing equipment recalibration recommended."
        ),
    },
    {
        # Near-duplicate of complaint 0 — same product/batch, different reporter.
        # Used to verify duplicate_detection triggers above threshold.
        "product_name": "Amoxicillin Trihydrate Capsules",
        "batch_lot_number": "BX-20240315-002",
        "customer_name": "PharmaDist UK Ltd",
        "complaint_date": "2024-03-18",
        "initial_severity": "Major",
        "priority": "High",
        "extraction_data": {
            "complaint_source": "Portal",
            "customer_name": "PharmaDist UK Ltd",
            "product_name": "Amoxicillin Trihydrate Capsules",
            "product_strength_grade": "500 mg",
            "batch_lot_number": "BX-20240315-002",
            "manufacturing_date": "2024-01-15",
            "expiry_date": "2026-01-14",
            "quantity_affected": "30 units",
            "complaint_type": "Packaging",
            "complaint_date": "2024-03-18",
            "detailed_complaint_description": (
                "30 bottles from batch BX-20240315-002 received with defective induction "
                "seals — the foil liner was not bonded to the bottle neck. Discovered at "
                "our Birmingham depot during goods-in inspection."
            ),
            "initial_severity": "Major",
            "priority": "High",
        },
        "risk_justification": "Same batch as CMP-0001; second report confirms systemic seal failure.",
        "root_cause_hypothesis": "Consistent with sealing machine temperature variance for this batch run.",
        "capa_text": "Batch already quarantined per CMP-0001 CAPA. Cross-reference both complaints.",
        "summary_text": (
            "Second Packaging complaint for Amoxicillin batch BX-20240315-002 from PharmaDist UK. "
            "30 units with defective seals — confirms systemic issue. Linked to CMP-0001."
        ),
    },
    {
        "product_name": "Metformin Hydrochloride Extended-Release Tablets",
        "batch_lot_number": "MF-2024-0089",
        "customer_name": "Dr. Priya Nair, City Hospital Mumbai",
        "complaint_date": "2024-04-22",
        "initial_severity": "Critical",
        "priority": "High",
        "extraction_data": {
            "complaint_source": "Email",
            "customer_name": "Dr. Priya Nair",
            "product_name": "Metformin Hydrochloride Extended-Release Tablets",
            "product_strength_grade": "1000 mg",
            "batch_lot_number": "MF-2024-0089",
            "manufacturing_date": "2024-02-10",
            "expiry_date": "2026-01-09",
            "quantity_affected": "90 tablets",
            "complaint_type": "Adverse Event",
            "complaint_date": "2024-04-22",
            "detailed_complaint_description": (
                "Three patients admitted with severe gastrointestinal distress including "
                "persistent vomiting, abdominal pain, and elevated lactate levels within "
                "6 hours of taking their regular Metformin 1000 mg dose. All three patients "
                "on stable long-term therapy with no prior adverse reactions."
            ),
            "initial_severity": "Critical",
            "priority": "High",
        },
        "risk_justification": (
            "Adverse Event with potential patient safety impact; three simultaneous cases "
            "from the same batch indicate a product quality issue rather than individual "
            "patient sensitivity. Critical severity per ICH E2A guidelines."
        ),
        "root_cause_hypothesis": (
            "Possible causes: super-potent tablet content uniformity failure, incorrect "
            "API concentration during granulation, or dissolution profile deviation causing "
            "dose dumping of the extended-release formulation."
        ),
        "capa_text": (
            "Corrective: Immediate batch recall MF-2024-0089; notify CDSCO per Schedule Y; "
            "retain reference samples for independent lab analysis. "
            "Preventive: Add content uniformity testing to in-process controls; "
            "validate dissolution method quarterly."
        ),
        "summary_text": (
            "Critical Adverse Event complaint for Metformin 1000mg ER (Batch MF-2024-0089): "
            "three patients hospitalised with GI distress post-dose. Immediate batch recall and "
            "CDSCO notification required."
        ),
    },
    {
        "product_name": "Atorvastatin Calcium Tablets",
        "batch_lot_number": "AT-2024-0441-B",
        "customer_name": "National Medicines Regulatory Authority (NMRA)",
        "complaint_date": "2024-06-05",
        "initial_severity": "Critical",
        "priority": "High",
        "extraction_data": {
            "complaint_source": "Regulatory",
            "customer_name": "National Medicines Regulatory Authority (NMRA)",
            "product_name": "Atorvastatin Calcium Tablets",
            "product_strength_grade": "40 mg",
            "batch_lot_number": "AT-2024-0441-B",
            "manufacturing_date": "2024-03-01",
            "expiry_date": "2027-02-28",
            "quantity_affected": "5000 tablets",
            "complaint_type": "Labeling",
            "complaint_date": "2024-06-05",
            "detailed_complaint_description": (
                "Patient information leaflet in batch AT-2024-0441-B instructs patients "
                "to take one tablet TWICE daily, contradicting the approved product "
                "monograph which specifies ONCE daily. Risk of patient overdose if "
                "PIL followed instead of prescriber instruction."
            ),
            "initial_severity": "Critical",
            "priority": "High",
        },
        "risk_justification": (
            "Incorrect PIL dosage instruction creates direct overdose risk for patients "
            "who follow the leaflet rather than their prescription. Class II recall trigger "
            "under local medicines regulations. Critical severity."
        ),
        "root_cause_hypothesis": (
            "Probable cause: incorrect master document version used during PIL print run "
            "for this batch; document change control process failure."
        ),
        "capa_text": (
            "Corrective: Class II recall of batch AT-2024-0441-B; notify prescribers and "
            "pharmacies; issue corrected PIL. "
            "Preventive: Implement mandatory second-check verification of PIL content "
            "against approved monograph before batch release; update SOP-DOC-012."
        ),
        "summary_text": (
            "Critical Labeling complaint from NMRA: Atorvastatin 40mg batch AT-2024-0441-B "
            "PIL instructs twice-daily dosing vs. approved once-daily. Class II recall initiated. "
            "Document control root cause identified."
        ),
    },
    {
        "product_name": "Ibuprofen Tablets",
        "batch_lot_number": "IBU-2024-0220",
        "customer_name": "Pharma Wholesale AG",
        "complaint_date": "2024-05-10",
        "initial_severity": "Minor",
        "priority": "Low",
        "extraction_data": {
            "complaint_source": "Distributor",
            "customer_name": "Pharma Wholesale AG",
            "product_name": "Ibuprofen Tablets",
            "product_strength_grade": "400 mg",
            "batch_lot_number": "IBU-2024-0220",
            "manufacturing_date": "2024-02-20",
            "expiry_date": "2026-02-19",
            "quantity_affected": "12 bottles",
            "complaint_type": "Delivery/Documentation",
            "complaint_date": "2024-05-10",
            "detailed_complaint_description": (
                "12 bottles in shipment received without the required Certificate of Analysis "
                "document. Product appears physically intact. Missing documentation prevents "
                "release to retail per local regulatory requirements."
            ),
            "initial_severity": "Minor",
            "priority": "Low",
        },
        "risk_justification": (
            "Documentation-only issue with no product quality or safety concern identified. "
            "Minor severity — product is likely releasable once CoA is resupplied."
        ),
        "root_cause_hypothesis": (
            "Probable cause: packing error during dispatch — CoA printed but not inserted "
            "into shipper box for this partial order."
        ),
        "capa_text": (
            "Corrective: Resend CoA documents to Pharma Wholesale AG within 24 hours. "
            "Preventive: Add CoA inclusion verification to dispatch checklist; "
            "implement packing scan confirmation for documentation items."
        ),
        "summary_text": (
            "Minor Delivery/Documentation complaint from Pharma Wholesale AG: 12 Ibuprofen "
            "400mg bottles received without Certificate of Analysis. No product quality concern. "
            "CoA to be resupplied within 24 hours."
        ),
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("Seeding complaints table...")

    async with get_async_session_factory()() as session:
        for i, data in enumerate(SEED_COMPLAINTS):
            # Generate the embedding for the description so duplicate_detection
            # has real vectors to compare against from day one.
            description: str = (
                data["extraction_data"].get("detailed_complaint_description") or ""
            )
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, embed_text, description
            )

            record = ComplaintORM(
                product_name=data["product_name"],
                batch_lot_number=data["batch_lot_number"],
                customer_name=data["customer_name"],
                complaint_date=data["complaint_date"],
                initial_severity=data["initial_severity"],
                priority=data["priority"],
                extraction_data=data["extraction_data"],
                risk_justification=data.get("risk_justification"),
                root_cause_hypothesis=data.get("root_cause_hypothesis"),
                capa_text=data.get("capa_text"),
                summary_text=data.get("summary_text"),
                description_embedding=embedding,
                raw_text=description,
                is_confirmed=True,
            )
            session.add(record)
            print(f"  [{i + 1}/5] {data['product_name']} — {data['batch_lot_number']}")

        await session.commit()

    print("\nSeed complete. 5 complaints inserted.")
    print("Near-duplicate pair: complaints 1 & 2 (same Amoxicillin batch)")
    print("Run 'python smoke_test_phase4.py' to verify duplicate detection.")


if __name__ == "__main__":
    asyncio.run(main())
