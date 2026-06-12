"""Unit tests for healthcare query routing."""

from __future__ import annotations

import pytest

from src.retrieval.query_router import (
    DEFAULT_CATEGORY_RULES,
    QueryCategory,
    QueryRouter,
    QueryRouterError,
)


@pytest.fixture
def router() -> QueryRouter:
    """Return a default query router instance."""
    return QueryRouter()


class TestQueryRouterCategories:
    """Tests for category detection."""

    def test_detect_workforce_category(self, router: QueryRouter) -> None:
        categories = router.detect_categories("doctor shortage in Bihar")
        assert QueryCategory.WORKFORCE in categories

    def test_detect_maternal_health_category(self, router: QueryRouter) -> None:
        categories = router.detect_categories("What is the maternal mortality rate?")
        assert QueryCategory.MATERNAL_HEALTH in categories

    def test_detect_disease_burden_category(self, router: QueryRouter) -> None:
        categories = router.detect_categories("Explain disease burden and DALY trends")
        assert QueryCategory.DISEASE_BURDEN in categories

    def test_detect_policy_category(self, router: QueryRouter) -> None:
        categories = router.detect_categories("Ayushman Bharat financing policy")
        assert QueryCategory.POLICY in categories

    def test_detect_multiple_categories(self, router: QueryRouter) -> None:
        categories = router.detect_categories("doctor staffing policy and budget")
        assert QueryCategory.WORKFORCE in categories
        assert QueryCategory.POLICY in categories

    def test_detect_no_category(self, router: QueryRouter) -> None:
        categories = router.detect_categories("general hospital information")
        assert categories == []

    def test_empty_query_raises(self, router: QueryRouter) -> None:
        with pytest.raises(ValueError, match="Query cannot be empty"):
            router.detect_categories("   ")


class TestQueryRouterSources:
    """Tests for preferred source routing."""

    def test_route_workforce_sources(self, router: QueryRouter) -> None:
        sources = router.route("nurse and pharmacist workforce vacancy")
        assert sources == ["rhs_2020.csv"]

    def test_route_maternal_health_sources(self, router: QueryRouter) -> None:
        sources = router.route("antenatal care and pregnancy outcomes")
        assert sources == ["NFHS-5_National_Report.pdf"]

    def test_route_disease_burden_sources(self, router: QueryRouter) -> None:
        sources = router.route("tuberculosis mortality and NCD disease burden")
        assert sources == [
            "2017_India_State_Level_Disease_Burden_Initiative_Full_Report.pdf"
        ]

    def test_route_policy_sources(self, router: QueryRouter) -> None:
        sources = router.route("national health policy budget financing")
        assert sources == [
            "National_Health_Policy_2017.pdf",
            "Ayushman_Bharat_Guidelines.pdf",
        ]

    def test_route_combined_sources_without_duplicates(self, router: QueryRouter) -> None:
        sources = router.route("doctor workforce policy budget")
        assert sources == [
            "rhs_2020.csv",
            "National_Health_Policy_2017.pdf",
            "Ayushman_Bharat_Guidelines.pdf",
        ]

    def test_route_unknown_query_returns_empty_list(self, router: QueryRouter) -> None:
        assert router.route("weather forecast") == []

    def test_get_sources_for_category(self, router: QueryRouter) -> None:
        sources = router.get_sources_for_category(QueryCategory.WORKFORCE)
        assert sources == ["rhs_2020.csv"]

    def test_get_sources_for_unknown_category_raises(self) -> None:
        custom_router = QueryRouter(rules=())
        with pytest.raises(QueryRouterError):
            custom_router.get_sources_for_category(QueryCategory.POLICY)


class TestQueryRouterConfiguration:
    """Tests for router configuration."""

    def test_default_rules_cover_all_categories(self) -> None:
        categories = {rule.category for rule in DEFAULT_CATEGORY_RULES}
        assert categories == set(QueryCategory)

    def test_keyword_matching_is_case_insensitive(self, router: QueryRouter) -> None:
        lower_sources = router.route("doctor shortage")
        upper_sources = router.route("DOCTOR SHORTAGE")
        assert lower_sources == upper_sources

    def test_phrase_keyword_matching(self, router: QueryRouter) -> None:
        assert router.route("Ayushman Bharat guidelines") == [
            "National_Health_Policy_2017.pdf",
            "Ayushman_Bharat_Guidelines.pdf",
        ]

    def test_word_boundary_matching_avoids_false_positive(self, router: QueryRouter) -> None:
        categories = router.detect_categories("recommended approach")
        assert QueryCategory.POLICY not in categories
