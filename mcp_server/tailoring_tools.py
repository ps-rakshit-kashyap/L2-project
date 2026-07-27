# mcp_server/tailoring_tools.py
# Implements two tailoring MCP tools:
#   1. rank_projects — Ranks resume projects by keyword relevance to the job.
#   2. tailor_resume — Generates a tailored resume with optimized summary,
#      skills, experience, and project sections based on job requirements.
#
# These are the final tools called by the agent, producing the output
# that the user sees in the results page.

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TailoringTools:
    """MCP tool implementations for resume tailoring and project ranking."""

    @staticmethod
    def rank_projects(
        projects: list,
        job_description: str,
    ) -> Dict[str, Any]:
        """Rank projects by relevance to the job description.
        
        For each project, computes a relevance score by counting
        word-level overlap between the project text (name + description
        + technologies) and the job description. Projects are sorted
        highest-to-lowest relevance.
        
        Args:
            projects: List of project dicts (each with name, description, technologies)
                      or plain strings from the parsed resume.
            job_description: Full text of the job posting.
            
        Returns:
            Dict with ranked_projects list (sorted by relevance descending).
        """
        logger.info("Ranking projects by relevance")
        jd_lower = job_description.lower()
        jd_words = set(jd_lower.split())

        scored: List[tuple] = []
        for project in projects:
            if isinstance(project, dict):
                # Concatenate project fields for text matching
                proj_text = (
                    f"{project.get('name', '')} "
                    f"{project.get('description', '')} "
                    f"{project.get('technologies', '')}"
                ).lower()
            else:
                proj_text = str(project).lower()

            # Count overlapping words between project and job description
            common = jd_words & set(proj_text.split())
            relevance = len(common)
            scored.append((project, relevance))

        # Sort by relevance score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [p[0] for p in scored]

        return {"ranked_projects": ranked}

    @staticmethod
    def tailor_resume(
        resume_data: Dict[str, Any],
        job_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a tailored version of the resume for a specific job.
        
        Creates a professional summary that highlights matched skills,
        and produces a tailored skills list that includes both matched
        skills and key missing skills (to suggest adding them).
        Experience and project sections are passed through as-is
        (they would be further refined by the LLM in a full implementation).
        
        Args:
            resume_data: Output from read_resume tool.
            job_data: Output from analyze_job_description tool.
            
        Returns:
            Dict with professional_summary, tailored_skills,
            tailored_experience, and tailored_projects.
        """
        logger.info("Tailoring resume content")
        resume_skills = resume_data.get("skills", [])
        job_skills = job_data.get("required_skills", [])

        # Case-insensitive comparison for skill matching
        resume_set = set(s.lower().strip() for s in resume_skills if s)
        job_set = set(s.lower().strip() for s in job_skills if s)

        matched = list(resume_set & job_set)
        missing = list(job_set - resume_set)

        # Build tailored skills: start with all matched skills (original casing),
        # then add up to 5 missing skills to suggest inclusion
        tailored_skills = list(dict.fromkeys(
            [s for s in resume_skills if s.lower().strip() in matched] +
            [s for s in job_skills if s.lower().strip() in missing][:5]
        ))

        # Generate a professional summary highlighting matched expertise
        professional_summary = (
            f"Experienced professional with expertise in "
            f"{', '.join(matched[:5]) if matched else 'software development'}. "
            f"Proven track record of delivering high-impact solutions. "
            f"Seeking to leverage these skills in a challenging new role."
        )

        return {
            "professional_summary": professional_summary,
            "tailored_skills": tailored_skills,
            "tailored_experience": resume_data.get("experience", []),
            "tailored_projects": resume_data.get("projects", []),
        }
