# utils/pdf_parser.py
# Handles parsing of PDF resume files using PyMuPDF (fitz).
# Extracts raw text from PDF and splits it into logical sections
# (summary, experience, education, skills, projects) using regex pattern matching.
# This is the foundation layer — MCP tools and the agent depend on this
# to understand the content of a resume.

import fitz
import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# SECTION_PATTERNS maps section names to regex patterns that detect
# common resume section headings. Each pattern captures the content
# between one heading and the next known heading (or end of string).
# The lookahead (?=...) allows matching without consuming the next heading.
SECTION_PATTERNS = {
    "summary": r"(?i)(professional\s*summary|summary|profile|about\s*me)[:\s]*(.*?)(?=\n\s*(?:experience|education|skills|projects|employment|work|certifications)|\Z)",
    "experience": r"(?i)(experience|employment|work\s*history|work\s*experience)[:\s]*(.*?)(?=\n\s*(?:education|skills|projects|certifications|summary)|\Z)",
    "education": r"(?i)(education|academic|qualifications)[:\s]*(.*?)(?=\n\s*(?:experience|skills|projects|certifications|summary)|\Z)",
    "skills": r"(?i)(skills|technical\s*skills|core\s*competencies|technologies)[:\s]*(.*?)(?=\n\s*(?:experience|education|projects|certifications|summary)|\Z)",
    "projects": r"(?i)(projects|personal\s*projects|key\s*projects)[:\s]*(.*?)(?=\n\s*(?:experience|education|skills|certifications|summary)|\Z)",
}


class PDFParser:
    """Parses PDF resumes and extracts structured data.
    
    Uses PyMuPDF (fitz) for PDF text extraction and regex-based
    section splitting to identify resume sections.
    """

    @staticmethod
    def extract_text(file_path: str) -> str:
        """Extract all text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            The complete text content of the PDF as a single string.
            
        Raises:
            ValueError: If the PDF cannot be opened or parsed.
        """
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise ValueError(f"Cannot parse PDF file: {e}")

    @staticmethod
    def parse_resume(file_path: str) -> Dict[str, Any]:
        """Parse a resume PDF into structured sections.
        
        Extracts text from the PDF, then attempts to split it into
        summary, skills, experience, education, and projects sections.
        The raw text is also preserved for further analysis (e.g. skill extraction).
        
        Args:
            file_path: Path to the resume PDF.
            
        Returns:
            Dict with keys: raw_text, summary, skills, experience, education, projects.
        """
        raw_text = PDFParser.extract_text(file_path)
        sections = PDFParser._split_sections(raw_text)

        return {
            "raw_text": raw_text,
            "summary": sections.get("summary", raw_text[:500]),
            "skills": sections.get("skills", []),
            "experience": sections.get("experience", []),
            "education": sections.get("education", []),
            "projects": sections.get("projects", []),
        }

    @staticmethod
    def _split_sections(text: str) -> Dict[str, Any]:
        """Split resume text into sections using regex patterns.
        
        Iterates over SECTION_PATTERNS and extracts content for each.
        For list-type sections (skills, projects, etc.), splits content
        into individual items by newline. Skills are further split by
        comma and pipe delimiters for fine-grained extraction.
        
        Args:
            text: Raw resume text.
            
        Returns:
            Dict mapping section names to their parsed content.
        """
        result = {
            "summary": "",
            "experience": [],
            "education": [],
            "skills": [],
            "projects": [],
        }

        for section_name, pattern in SECTION_PATTERNS.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(2).strip()
                # For list-type sections, split into lines and clean up
                if section_name in ("skills", "projects", "experience", "education"):
                    items = [line.strip("- *").strip()
                             for line in content.split("\n")
                             if line.strip() and len(line.strip()) > 3]
                    # Skills need additional splitting on commas and pipes
                    if section_name in ("skills",):
                        skill_items = []
                        for item in items:
                            skill_items.extend([s.strip() for s in re.split(r"[,|/]", item) if s.strip()])
                        result[section_name] = skill_items
                    else:
                        result[section_name] = items[:15]
                else:
                    result[section_name] = content

        # Fallback: if no sections were detected, use first 10 lines as summary
        if not any(result.values()):
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            result["summary"] = " ".join(lines[:10])

        return result
