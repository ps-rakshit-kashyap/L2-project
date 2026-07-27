# utils/skill_extractor.py
# Provides skill extraction from text content using keyword matching.
# Maintains a comprehensive list of technical and professional skills
# commonly found in tech resumes. The extractor performs case-insensitive
# regex matching against this keyword list.
#
# Used by multiple MCP tools (read_resume, analyze_job_description)
# to identify skills present in resumes and job descriptions.

import re
from typing import List, Set

# Comprehensive list of skill keywords covering:
# - Programming languages (Python, Java, C++, etc.)
# - Frameworks & libraries (React, Django, TensorFlow, etc.)
# - Databases & storage (SQL, MongoDB, Redis, etc.)
# - Cloud & DevOps (AWS, Docker, Kubernetes, etc.)
# - Tools & platforms (Git, Jenkins, JIRA, etc.)
# - Methodologies (Agile, Scrum, TDD, etc.)
# - Domains (ML, Data Science, NLP, etc.)
# Each entry is a discrete skill name used for exact matching.
SKILL_KEYWORDS: List[str] = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Ruby", "Go", "Rust",
    "React", "Angular", "Vue.js", "Vue", "Svelte", "Next.js", "Nuxt",
    "Node.js", "Deno", "Bun",
    "Django", "Flask", "FastAPI", "Spring Boot", "Express", "ASP.NET",
    "SQL", "MongoDB", "PostgreSQL", "MySQL", "SQLite", "Redis", "Elasticsearch",
    "AWS", "Azure", "GCP", "Cloud",
    "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins",
    "Git", "CI/CD", "GitHub Actions", "GitLab CI",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "LLM", "RAG",
    "TensorFlow", "PyTorch", "Scikit-learn", "Keras", "XGBoost",
    "REST API", "GraphQL", "gRPC", "WebSocket",
    "Agile", "Scrum", "JIRA", "Confluence",
    "Leadership", "Communication", "Team Management", "Project Management",
    "Data Science", "Data Engineering", "Data Analysis", "Data Pipeline",
    "Spark", "Hadoop", "Kafka", "Airflow", "Snowflake", "BigQuery",
    "Linux", "Bash", "PowerShell", "Shell Scripting",
    "HTML", "CSS", "SASS", "Tailwind CSS", "Bootstrap",
    "OOP", "Design Patterns", "Microservices", "System Design",
    "Testing", "Unit Testing", "Integration Testing", "TDD",
    "Django REST", "Celery", "RabbitMQ", "Nginx",
    "Postman", "Swagger", "OpenAPI",
    "Selenium", "Cypress", "Playwright",
    "Tableau", "Power BI", "Looker",
    "Figma", "Adobe XD", "Sketch",
    "Jest", "Mocha", "Chai", "Pytest", "JUnit",
    "Webpack", "Vite", "Parcel", "Babel",
    "Redux", "Zustand", "Pinia", "Vuex",
    "Three.js", "D3.js", "Chart.js",
    "R", "MATLAB", "Scala", "Kotlin", "Swift",
    "Firebase", "Supabase", "Hasura",
    "Stripe", "PayPal API",
    "OAuth", "JWT", "SSL/TLS",
    "SOLID", "Clean Architecture", "DDD", "CQRS", "Event Sourcing",
]


class SkillExtractor:
    """Extracts skill keywords from text content.
    
    Uses case-insensitive regex matching against a curated list of
    technology and professional skills commonly found in tech resumes.
    Results are deduplicated and sorted by length (longest first)
    to prioritize more specific skill names.
    """

    @staticmethod
    def extract_skills(text: str) -> List[str]:
        """Extract all matching skills from the given text.
        
        Case-insensitive regex search for each skill keyword.
        Longer matches are returned first to avoid substring conflicts
        (e.g. 'React' before 'R').
        
        Args:
            text: Arbitrary text content (resume text, job description, etc.).
            
        Returns:
            Deduplicated, sorted list of matched skill names.
        """
        found: Set[str] = set()
        text_lower = text.lower()

        for skill in SKILL_KEYWORDS:
            # Use re.escape to handle special characters in skill names
            escaped = re.escape(skill)
            if re.search(escaped, text, re.IGNORECASE):
                found.add(skill)

        # Sort by length descending so more specific skills appear first
        return sorted(found, key=lambda s: len(s), reverse=True)

    @staticmethod
    def extract_skills_from_list(items: List[str]) -> List[str]:
        """Extract skills from a list of text strings.
        
        Convenience method that joins list items and runs extraction.
        
        Args:
            items: List of text strings (e.g. project descriptions).
            
        Returns:
            Deduplicated, sorted list of matched skill names.
        """
        all_text = " ".join(items)
        return SkillExtractor.extract_skills(all_text)
