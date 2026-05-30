import json
import shutil
import os
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import UserResponse, ProjectCreate, ProjectResponse
from app.services.template_engine import generate_project, list_templates

router = APIRouter(prefix="/api/projects", tags=["Projects"])

@router.get("/", response_model=List[ProjectResponse])
async def list_projects(current_user: UserResponse = Depends(get_current_user)):
    """List all projects for the current user."""
    projects = []
    async with get_db() as db:
        async with db.execute("SELECT * FROM projects WHERE user_id = ? ORDER BY created_at DESC", (current_user.id,)) as cursor:
            async for row in cursor:
                projects.append(ProjectResponse(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    template=row["template"],
                    config_json=row["config_json"],
                    status=row["status"],
                    created_at=row["created_at"]
                ))
    return projects

@router.post("/create")
async def create_project(req: ProjectCreate, current_user: UserResponse = Depends(get_current_user)):
    """Generate a new AI project and save it to the database."""
    # Ensure template exists
    templates = list_templates()
    if req.template not in [t["id"] for t in templates]:
        raise HTTPException(status_code=400, detail=f"Template '{req.template}' not found")
        
    config = json.loads(req.config_json) if req.config_json else {}
    config["project_name"] = req.name
    
    # Generate the actual project code
    result = await generate_project(
        template_id=req.template,
        tier=current_user.tier,
        config=config
    )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    # Save to database
    async with get_db() as db:
        cursor = await db.execute(
            "INSERT INTO projects (user_id, name, template, config_json, status) VALUES (?, ?, ?, ?, ?)",
            (current_user.id, req.name, req.template, req.config_json, "completed")
        )
        await db.commit()
        project_id = cursor.lastrowid
        
        # Return the created project
        async with db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)) as fetch_cursor:
            row = await fetch_cursor.fetchone()
            return {
                "project": ProjectResponse(
                    id=row["id"],
                    user_id=row["user_id"],
                    name=row["name"],
                    template=row["template"],
                    config_json=row["config_json"],
                    status=row["status"],
                    created_at=row["created_at"]
                ),
                "generation_result": result
            }

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: int, current_user: UserResponse = Depends(get_current_user)):
    """Get details of a specific project."""
    async with get_db() as db:
        async with db.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, current_user.id)) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Project not found")
            return ProjectResponse(
                id=row["id"],
                user_id=row["user_id"],
                name=row["name"],
                template=row["template"],
                config_json=row["config_json"],
                status=row["status"],
                created_at=row["created_at"]
            )

@router.delete("/{project_id}")
async def delete_project(project_id: int, current_user: UserResponse = Depends(get_current_user)):
    """Delete a project and its generated files."""
    async with get_db() as db:
        async with db.execute("SELECT * FROM projects WHERE id = ? AND user_id = ?", (project_id, current_user.id)) as cursor:
            row = await cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Project not found")
                
        # Attempt to delete the directory
        # The default generation directory is 'projects/<project_name>'
        try:
            project_dir = Path("projects") / row["name"]
            if project_dir.exists() and project_dir.is_dir():
                shutil.rmtree(project_dir)
        except Exception as e:
            # We still delete the DB record even if file deletion fails
            pass
            
        await db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        await db.commit()
        
    return {"status": "success", "message": "Project deleted"}
