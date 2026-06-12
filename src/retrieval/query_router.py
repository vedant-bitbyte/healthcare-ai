"""Keyword-based query routing for healthcare RAG retrieval."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryCategory(str, Enum):
    """Supported healthcare query categories."""

    WORKFORCE = "workforce"
    MATERNAL_HEALTH = "maternal_health"
    DISEASE_BURDEN = "disease_burden"
    POLICY = "policy"


@dataclass(frozen=True)
class CategoryRule:
    """Routing rule mapping keywords to preferred source documents."""

    category: QueryCategory
    keywords: tuple[str, ...]
    sources: tuple[str, ...]


DEFAULT_CATEGORY_RULES: tuple[CategoryRule, ...] = (
    CategoryRule(
        category=QueryCategory.WORKFORCE,
        keywords=(
            "doctor",
            "doctors",
            "specialist",
            "specialists",
            "nurse",
            "nurses",
            "pharmacist",
            "pharmacists",
            "workforce",
            "staffing",
            "vacancy",
            "vacancies",
            "shortage",
        ),
        sources=("rhs_2020.csv",),
    ),
    CategoryRule(
        category=QueryCategory.MATERNAL_HEALTH,
        keywords=(
            "maternal",
            "pregnancy",
            "antenatal",
            "delivery",
            "mmr",
        ),
        sources=("NFHS-5_National_Report.pdf",),
    ),
    CategoryRule(
        category=QueryCategory.DISEASE_BURDEN,
        keywords=(
            "disease burden",
            "daly",
            "ncd",
            "tuberculosis",
        ),
        sources=("2017_India_State_Level_Disease_Burden_Initiative_Full_Report.pdf",),
    ),
    CategoryRule(
        category=QueryCategory.POLICY,
        keywords=(
            "policy",
            "budget",
            "financing",
            "ayushman bharat",
        ),
        sources=(
            "National_Health_Policy_2017.pdf",
            "Ayushman_Bharat_Guidelines.pdf",
        ),
    ),
)


class QueryRouterError(Exception):
    """Raised when query routing fails."""


class QueryRouter:
    """Route user queries to preferred healthcare source documents."""

    def __init__(self, rules: tuple[CategoryRule, ...] = DEFAULT_CATEGORY_RULES) -> None:
        """Initialize the router with category rules.

        Args:
            rules: Tuple of category rules used for keyword matching.
        """
        self._rules = rules
        self._rules_by_category = {rule.category: rule for rule in rules}

    @property
    def rules(self) -> tuple[CategoryRule, ...]:
        """Return configured routing rules."""
        return self._rules

    def detect_categories(self, query: str) -> list[QueryCategory]:
        """Detect query categories using keyword matching.

        Args:
            query: User query text.

        Returns:
            Matched categories in rule definition order.

        Raises:
            ValueError: If the query is empty.
        """
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            raise ValueError("Query cannot be empty")

        matched_categories: list[QueryCategory] = []

        for rule in self._rules:
            if self._matches_rule(normalized_query, rule):
                matched_categories.append(rule.category)
                logger.debug("Query matched category '%s'", rule.category.value)

        if matched_categories:
            logger.info(
                "Detected categories: %s",
                ", ".join(category.value for category in matched_categories),
            )
        else:
            logger.info("No query category detected")

        return matched_categories

    def route(self, query: str) -> list[str]:
        """Return preferred source files for a query.

        Args:
            query: User query text.

        Returns:
            De-duplicated list of preferred source filenames. Returns an empty
            list when no category matches.

        Raises:
            ValueError: If the query is empty.
        """
        categories = self.detect_categories(query)
        preferred_sources: list[str] = []
        seen: set[str] = set()

        for category in categories:
            rule = self._rules_by_category[category]
            for source in rule.sources:
                if source not in seen:
                    seen.add(source)
                    preferred_sources.append(source)

        logger.info("Preferred sources: %s", preferred_sources or "none")
        return preferred_sources

    def get_sources_for_category(self, category: QueryCategory) -> list[str]:
        """Return preferred sources for a specific category.

        Args:
            category: Query category enum value.

        Returns:
            Preferred source filenames for the category.

        Raises:
            QueryRouterError: If the category is not configured.
        """
        rule = self._rules_by_category.get(category)
        if rule is None:
            raise QueryRouterError(f"No routing rule configured for category '{category.value}'")

        return list(rule.sources)

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Normalize query text for consistent keyword matching."""
        return " ".join(query.lower().split())

    @staticmethod
    def _matches_keyword(normalized_query: str, keyword: str) -> bool:
        """Check whether a keyword matches the normalized query."""
        normalized_keyword = keyword.lower().strip()
        if not normalized_keyword:
            return False

        if " " in normalized_keyword:
            return normalized_keyword in normalized_query

        return re.search(rf"\b{re.escape(normalized_keyword)}\b", normalized_query) is not None

    def _matches_rule(self, normalized_query: str, rule: CategoryRule) -> bool:
        """Return True if any keyword in the rule matches the query."""
        return any(
            self._matches_keyword(normalized_query, keyword)
            for keyword in rule.keywords
        )
