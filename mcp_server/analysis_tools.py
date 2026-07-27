# mcp_server/analysis_tools.py
# Implements three analysis MCP tools:
#   1. analyze_job_description — Extracts role, required/preferred skills, and keywords from a JD.
#   2. skill_gap_analysis — Compares resume skills against job requirements,
#      producing matched/missing lists and a percentage score.
#   3. ats_score — Calculates ATS compatibility score with actionable recommendations.
#
# These tools form the "analysis" phase of the agent's workflow,
# helping the agent understand the gap between the resume and job requirements.

from typing import Dict, Any, List
from utils.skill_extractor import SkillExtractor
from utils.ats import ATSScorer
import logging

logger = logging.getLogger(__name__)


class AnalysisTools:
    """MCP tool implementations for job analysis and skill comparison."""

    @staticmethod
    def analyze_job_description(job_description: str) -> Dict[str, Any]:
        """Extract skill requirements from a job description.
        
        Uses SkillExtractor to identify all technical/professional skills
        mentioned in the job posting. Also produces lowercase keywords
        for downstream matching.
        
        Args:
            job_description: Full text of the job posting.
            
        Returns:
            Dict with role, required_skills, preferred_skills, keywords.
        """
        logger.info("Analyzing job description")
        skills = SkillExtractor.extract_skills(job_description)

        # Lowercase keywords for case-insensitive comparison later
        keywords = [s.lower() for s in skills]

        return {
            "role": "Detected from job description",
            "required_skills": skills,
            "preferred_skills": [],  # Not yet distinguished from required
            "keywords": keywords,
        }

    @staticmethod
    def skill_gap_analysis(
        resume_data: Dict[str, Any],
        job_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare resume skills against job requirements.
        
        Performs case-insensitive set comparison between resume skills
        and job-required skills. Returns:
        - matched_skills: Skills that appear in both.
        - missing_skills: Skills required by job but absent from resume.
        - score: Percentage of required skills that are matched.
        
        Args:
            resume_data: Output from read_resume tool.
            job_data: Output from analyze_job_description tool.
            
        Returns:
            Dict with matched_skills, missing_skills, and score.
        """
        logger.info("Performing skill gap analysis")
        resume_skills = [s.lower().strip() for s in resume_data.get("skills", []) if s]
        job_skills = [s.lower().strip() for s in job_data.get("required_skills", []) if s]

        resume_set = set(resume_skills)
        job_set = set(job_skills)

        matched = sorted(resume_set & job_set)
        missing = sorted(job_set - resume_set)

        # Calculate percentage, guarding against division by zero
        score = int((len(matched) / max(len(job_set), 1)) * 100)
        score = min(score, 100)

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "score": score,
        }

    @staticmethod
    def ats_score(
        resume_data: Dict[str, Any],
        job_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate ATS compatibility score with recommendations.
        
        Delegates scoring to ATSScorer.calculate_score and generates
        actionable recommendations via ATSScorer.get_recommendations.
        
        Args:
            resume_data: Output from read_resume tool.
            job_data: Output from analyze_job_description tool.
            
        Returns:
            Dict with score (0-100) and list of recommendations.
        """
        logger.info("Calculating ATS score")
        resume_skills = resume_data.get("skills", [])
        job_skills = job_data.get("required_skills", [])

        score = ATSScorer.calculate_score(resume_skills, job_skills)

        # Compute matched/missing for recommendation generation
        resume_set = set(s.lower().strip() for s in resume_skills if s)
        job_set = set(s.lower().strip() for s in job_skills if s)
        matched = sorted(resume_set & job_set)
        missing = sorted(job_set - resume_set)

        recommendations = ATSScorer.get_recommendations(
            matched, missing, resume_data, job_data
        )

        return {
            "score": score,
            "recommendations": recommendations,
        }
