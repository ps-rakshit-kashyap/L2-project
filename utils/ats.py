# utils/ats.py
# Implements ATS (Applicant Tracking System) compatibility scoring
# and recommendation generation. ATS software scans resumes for
# keywords matching the job description. This module provides:
#   - calculate_score: Simple ratio of matched skills to required skills
#   - get_recommendations: Actionable tips to improve ATS compatibility
#
# Used by the ats_score MCP tool in analysis_tools.py.

from typing import List, Dict, Any, Tuple


class ATSScorer:
    """Calculates ATS compatibility scores and generates recommendations.
    
    The score is computed as the percentage of job-required skills that
    are present in the resume. Recommendations provide actionable advice
    for improving ATS performance.
    """

    @staticmethod
    def calculate_score(resume_skills: List[str], job_skills: List[str]) -> int:
        """Calculate ATS match percentage.
        
        Compares resume skills against required job skills using
        case-insensitive set intersection. Returns an integer 0-100.
        
        Args:
            resume_skills: Skills detected in the resume.
            job_skills: Skills required by the job description.
            
        Returns:
            Integer percentage of job skills found in resume.
        """
        if not job_skills:
            return 0
        resume_set = set(s.lower().strip() for s in resume_skills if s)
        job_set = set(s.lower().strip() for s in job_skills if s)
        if not job_set:
            return 0
        matched = resume_set & job_set
        score = int((len(matched) / len(job_set)) * 100)
        return min(score, 100)

    @staticmethod
    def get_recommendations(
        matched: List[str],
        missing: List[str],
        resume_data: Dict[str, Any],
        job_data: Dict[str, Any],
    ) -> List[str]:
        """Generate ATS optimization recommendations.
        
        Produces actionable tips based on the gap analysis results.
        Includes specific missing skills to add, plus general best practices
        for ATS-friendly resume formatting.
        
        Args:
            matched: Skills present in both resume and job requirements.
            missing: Skills required by job but missing from resume.
            resume_data: Full parsed resume data (for context).
            job_data: Full parsed job data (for context).
            
        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        # Specific suggestion: add missing keywords
        if missing:
            top_missing = missing[:5]
            recommendations.append(
                f"Add missing keywords to your resume: {', '.join(top_missing)}"
            )

        # If few skills matched, suggest keyword density improvement
        if len(matched) < 3:
            recommendations.append(
                "Increase keyword density — incorporate more job-specific terms into your experience descriptions"
            )

        # General ATS best practices
        recommendations.append("Use strong action verbs (e.g., 'Developed', 'Architected', 'Optimized')")
        recommendations.append("Quantify achievements with numbers (e.g., 'Reduced latency by 40%')")
        recommendations.append("Ensure consistent formatting, fonts, and bullet styles")
        recommendations.append("Keep resume to 1–2 pages for best ATS compatibility")
        recommendations.append("Use standard section headings (Experience, Education, Skills, Projects)")

        return recommendations
