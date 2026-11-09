from fastapi import APIRouter, HTTPException, status
from app.database import participants_col, samples_col
from app.models.participant import Participant, ParticipantCreate
from datetime import datetime, timezone

router = APIRouter(prefix="/api/participants", tags=["Participants"])

@router.get("")
async def list_participants():
    """List all registered participants with real sample contribution counts."""
    cursor = participants_col().find().sort("participant_id", 1)
    participants_list = await cursor.to_list(length=500)
    
    result = []
    for p in participants_list:
        p_id = p["participant_id"]
        samples_count = await samples_col().count_documents({"participant_id": p_id})
        
        p_data = dict(p)
        p_data.pop("_id", None)
        p_data["samples_count"] = samples_count
        result.append(p_data)
        
    return {"participants": result, "total": len(result)}

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_participant(payload: ParticipantCreate):
    """Register a new dataset collection participant with consent status."""
    p_id = payload.participant_id.strip()
    existing = await participants_col().find_one({"participant_id": p_id})
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Participant ID '{p_id}' already exists."
        )
        
    participant = Participant(
        participant_id=p_id,
        display_name=payload.display_name.strip(),
        consent_status=payload.consent_status,
        notes=payload.notes.strip()
    )
    
    doc = participant.model_dump()
    await participants_col().insert_one(doc)
    doc.pop("_id", None)
    return doc

@router.get("/{participant_id}")
async def get_participant(participant_id: str):
    doc = await participants_col().find_one({"participant_id": participant_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    doc.pop("_id", None)
    doc["samples_count"] = await samples_col().count_documents({"participant_id": participant_id})
    return doc

@router.delete("/{participant_id}")
async def delete_participant(participant_id: str):
    """Deletes participant if they have no active dataset samples or explicit removal."""
    doc = await participants_col().find_one({"participant_id": participant_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Participant not found")
        
    samples_count = await samples_col().count_documents({"participant_id": participant_id})
    if samples_count > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot delete participant '{participant_id}' because {samples_count} dataset samples are associated with them. Delete their samples first."
        )
        
    await participants_col().delete_one({"participant_id": participant_id})
    return {"success": True, "message": f"Participant '{participant_id}' deleted."}
