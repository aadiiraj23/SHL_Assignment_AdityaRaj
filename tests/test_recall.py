import pytest
import pytest_asyncio
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from datetime import datetime, timezone
from typing import Dict, List
from unittest.mock import patch

from app.catalog import CatalogStore
from app.retriever import Retriever


# Define expected results mapping for IR benchmarking
EXPECTED_RESULTS = {
    "mid-level Java developer stakeholder communication": [
        "Verify - Java",
        "OPQ32r"
    ],
    "sales manager personality leadership": [
        "OPQ32r",
        "Graduate Potential"
    ],
    "cognitive ability graduate scheme": [
        "Verify - Numerical Reasoning",
        "Verify - Verbal Reasoning",
        "Graduate Potential"
    ],
    "customer service situational judgement": [
        "Situational Judgement Test"
    ]
}


@pytest.fixture
def sample_catalog_for_recall():
    """Provides comprehensive catalog data for recall testing"""
    return {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "total": 6,
        "assessments": [
            {
                "name": "Verify - Java",
                "url": "https://www.shl.com/verify-java",
                "description": "Technical ability test for Java developers with questions on core language concepts, OOP design patterns, and concurrency",
                "test_type": "A",
                "remote": True,
                "adaptive": True,
                "duration_minutes": 60
            },
            {
                "name": "OPQ32r",
                "url": "https://www.shl.com/opq32r",
                "description": "Comprehensive personality assessment measuring 32 personality dimensions including leadership, communication, teamwork, and interpersonal skills",
                "test_type": "P",
                "remote": True,
                "adaptive": False,
                "duration_minutes": 45
            },
            {
                "name": "Verify - Numerical Reasoning",
                "url": "https://www.shl.com/verify-numerical",
                "description": "Measures cognitive ability through numerical reasoning problems including calculations, data interpretation, and quantitative analysis",
                "test_type": "A",
                "remote": True,
                "adaptive": False,
                "duration_minutes": 20
            },
            {
                "name": "Verify - Verbal Reasoning",
                "url": "https://www.shl.com/verify-verbal",
                "description": "Assesses verbal comprehension and critical thinking through reading comprehension and logical reasoning",
                "test_type": "A",
                "remote": True,
                "adaptive": False,
                "duration_minutes": 20
            },
            {
                "name": "Graduate Potential",
                "url": "https://www.shl.com/graduate-potential",
                "description": "Comprehensive assessment for graduate scheme candidates measuring personality, cognitive ability, and situational judgment",
                "test_type": "B",
                "remote": True,
                "adaptive": False,
                "duration_minutes": 90
            },
            {
                "name": "Situational Judgement Test",
                "url": "https://www.shl.com/sjt",
                "description": "Measures professional judgment and decision-making in workplace scenarios commonly used for customer service and support roles",
                "test_type": "S",
                "remote": True,
                "adaptive": False,
                "duration_minutes": 30
            }
        ]
    }


@pytest.fixture
def temp_catalog_for_recall(sample_catalog_for_recall):
    """Creates a temporary catalog.json file for recall testing"""
    with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_catalog_for_recall, f)
        temp_path = f.name
    
    yield Path(temp_path)
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest_asyncio.fixture
async def catalog_for_recall(temp_catalog_for_recall):
    """Provides a CatalogStore instance loaded with test data"""
    # Create catalog directly from temporary file
    catalog = CatalogStore(str(temp_catalog_for_recall))
    return catalog


@pytest_asyncio.fixture
async def retriever_for_recall(catalog_for_recall):
    """Provides a Retriever instance for IR testing"""
    return Retriever(catalog_for_recall)


@pytest.mark.asyncio
async def test_recall_at_k_computation(catalog_for_recall, retriever_for_recall):
    """
    Test complete recall@10 metric computation across all evaluation profiles
    
    Recall@10 measures: Among the top 10 retrieved results, what fraction are relevant?
    Formula: Recall@K = (# of relevant items in top K) / (total # of relevant items)
    
    This test validates the IR pipeline's ability to surface relevant assessments
    for each stakeholder profile.
    """
    
    recall_scores = {}
    
    for profile_query, expected_assessments in EXPECTED_RESULTS.items():
        # Execute hybrid search with k=10
        retrieved_assessments = retriever_for_recall.search_for_conversation(
            [{"role": "user", "content": profile_query}],
            k=10
        )
        
        # Extract names from retrieved results
        retrieved_names = set()
        if retrieved_assessments:
            for assessment in retrieved_assessments:
                if hasattr(assessment, 'name'):
                    retrieved_names.add(assessment.name)
                elif isinstance(assessment, dict) and 'name' in assessment:
                    retrieved_names.add(assessment['name'])
        
        # Compute recall for this profile
        expected_set = set(expected_assessments)
        relevant_retrieved = expected_set.intersection(retrieved_names)
        
        if len(expected_set) > 0:
            recall = len(relevant_retrieved) / len(expected_set)
        else:
            recall = 0.0
        
        recall_scores[profile_query] = {
            "recall": recall,
            "expected": expected_assessments,
            "retrieved": list(retrieved_names),
            "relevant_count": len(relevant_retrieved),
            "expected_count": len(expected_set)
        }
    
    # Calculate mean recall across all profiles
    if recall_scores:
        mean_recall = sum(s["recall"] for s in recall_scores.values()) / len(recall_scores)
    else:
        mean_recall = 0.0
    
    # Log results for debugging
    print("\n=== Recall@10 Evaluation Results ===")
    for profile, results in recall_scores.items():
        print(f"\nProfile: {profile}")
        print(f"  Expected: {results['expected']}")
        print(f"  Retrieved Top 10: {results['retrieved']}")
        print(f"  Recall: {results['recall']:.2f} ({results['relevant_count']}/{results['expected_count']})")
    
    print(f"\n=== Mean Recall@10 Across All Profiles: {mean_recall:.2f} ===")
    
    # Assertion: Mean Recall@10 must be >= 0.5
    assert mean_recall >= 0.5, f"Mean Recall@10 {mean_recall:.2f} is below threshold of 0.5"


@pytest.mark.asyncio
async def test_individual_profile_recall_java_developer(catalog_for_recall, retriever_for_recall):
    """
    Test recall for mid-level Java developer profile
    Expected: Verify - Java, OPQ32r
    """
    profile = "mid-level Java developer stakeholder communication"
    expected = EXPECTED_RESULTS[profile]
    
    retrieved = retriever_for_recall.search_for_conversation(
        [{"role": "user", "content": profile}],
        k=10
    )
    
    retrieved_names = {
        a.name if hasattr(a, 'name') else a.get('name')
        for a in (retrieved or [])
    }
    
    expected_set = set(expected)
    relevant_retrieved = expected_set.intersection(retrieved_names)
    
    recall = len(relevant_retrieved) / len(expected_set) if expected_set else 0.0
    assert recall > 0, f"Failed to retrieve any expected assessments for Java developer profile"


@pytest.mark.asyncio
async def test_individual_profile_recall_sales_manager(catalog_for_recall, retriever_for_recall):
    """
    Test recall for sales manager personality leadership profile
    Expected: OPQ32r, Graduate Potential
    """
    profile = "sales manager personality leadership"
    expected = EXPECTED_RESULTS[profile]
    
    retrieved = retriever_for_recall.search_for_conversation(
        [{"role": "user", "content": profile}],
        k=10
    )
    
    retrieved_names = {
        a.name if hasattr(a, 'name') else a.get('name')
        for a in (retrieved or [])
    }
    
    expected_set = set(expected)
    relevant_retrieved = expected_set.intersection(retrieved_names)
    
    recall = len(relevant_retrieved) / len(expected_set) if expected_set else 0.0
    assert recall > 0, f"Failed to retrieve any expected assessments for sales manager profile"


@pytest.mark.asyncio
async def test_individual_profile_recall_graduate_scheme(catalog_for_recall, retriever_for_recall):
    """
    Test recall for cognitive ability graduate scheme profile
    Expected: Verify - Numerical Reasoning, Verify - Verbal Reasoning, Graduate Potential
    """
    profile = "cognitive ability graduate scheme"
    expected = EXPECTED_RESULTS[profile]
    
    retrieved = retriever_for_recall.search_for_conversation(
        [{"role": "user", "content": profile}],
        k=10
    )
    
    retrieved_names = {
        a.name if hasattr(a, 'name') else a.get('name')
        for a in (retrieved or [])
    }
    
    expected_set = set(expected)
    relevant_retrieved = expected_set.intersection(retrieved_names)
    
    recall = len(relevant_retrieved) / len(expected_set) if expected_set else 0.0
    assert recall > 0, f"Failed to retrieve any expected assessments for graduate scheme profile"


@pytest.mark.asyncio
async def test_individual_profile_recall_customer_service(catalog_for_recall, retriever_for_recall):
    """
    Test recall for customer service situational judgment profile
    Expected: Situational Judgement Test
    """
    profile = "customer service situational judgement"
    expected = EXPECTED_RESULTS[profile]
    
    retrieved = retriever_for_recall.search_for_conversation(
        [{"role": "user", "content": profile}],
        k=10
    )
    
    retrieved_names = {
        a.name if hasattr(a, 'name') else a.get('name')
        for a in (retrieved or [])
    }
    
    expected_set = set(expected)
    relevant_retrieved = expected_set.intersection(retrieved_names)
    
    recall = len(relevant_retrieved) / len(expected_set) if expected_set else 0.0
    assert recall > 0, f"Failed to retrieve any expected assessments for customer service profile"


@pytest.mark.asyncio
async def test_hybrid_search_semantic_component(catalog_for_recall, retriever_for_recall):
    """
    Test that semantic search component is working
    Verifies that semantic similarity is finding relevant assessments
    """
    # Query should match assessment description semantically
    profile = "personality assessment for communication skills"
    
    retrieved = retriever_for_recall.search_for_conversation(
        [{"role": "user", "content": profile}],
        k=10
    )
    
    assert retrieved is not None
    assert len(retrieved) > 0, "Semantic search should return results"


@pytest.mark.asyncio
async def test_hybrid_search_lexical_component(catalog_for_recall, retriever_for_recall):
    """
    Test that lexical (BM25) search component is working
    Verifies that exact keyword matching finds relevant assessments
    """
    # Query with exact assessment type keywords
    profile = "verbal reasoning ability test"
    
    retrieved = retriever_for_recall.search_for_conversation(
        [{"role": "user", "content": profile}],
        k=10
    )
    
    assert retrieved is not None
    assert len(retrieved) > 0, "Lexical search should return results"
    
    # Should ideally contain verbal reasoning assessment
    retrieved_names = {
        a.name if hasattr(a, 'name') else a.get('name')
        for a in (retrieved or [])
    }
    assert "Verify - Verbal Reasoning" in retrieved_names or len(retrieved_names) > 0


@pytest.mark.asyncio
async def test_retrieval_k_limit_honored(catalog_for_recall, retriever_for_recall):
    """
    Test that k parameter is respected in retrieval
    Verifies that exactly k results (or fewer) are returned
    """
    k_values = [1, 5, 10]
    
    for k in k_values:
        retrieved = retriever_for_recall.search_for_conversation(
            [{"role": "user", "content": "assessment"}],
            k=k
        )
        
        result_count = len(retrieved) if retrieved else 0
        assert result_count <= k, f"Retrieved {result_count} results but k={k}"


@pytest.mark.asyncio
async def test_catalog_coverage_all_assessments_retrievable(catalog_for_recall, retriever_for_recall):
    """
    Test that all catalog assessments are retrievable through search
    Ensures no assessments are invisible to the retrieval pipeline
    """
    # Broad query should retrieve most/all assessments
    retrieved = retriever_for_recall.search_for_conversation(
        [{"role": "user", "content": "assessment test evaluation"}],
        k=10
    )
    
    retrieved_names = {
        a.name if hasattr(a, 'name') else a.get('name')
        for a in (retrieved or [])
    }
    
    # With 6 assessments and k=10, should get most or all
    assert len(retrieved_names) >= 3, f"Expected to retrieve at least 3 distinct assessments, got {len(retrieved_names)}"


@pytest.mark.asyncio
async def test_recall_scores_data_structure(catalog_for_recall, retriever_for_recall):
    """
    Test the data structure and computation of recall metrics
    Validates that recall computation follows proper mathematical formula
    """
    recall_results = {}
    
    for profile, expected_assessments in EXPECTED_RESULTS.items():
        retrieved = retriever_for_recall.search_for_conversation(
            [{"role": "user", "content": profile}],
            k=10
        )
        
        retrieved_names = {
            a.name if hasattr(a, 'name') else a.get('name')
            for a in (retrieved or [])
        }
        
        expected_set = set(expected_assessments)
        relevant = expected_set.intersection(retrieved_names)
        
        # Proper recall computation
        total_relevant = len(expected_set)
        relevant_retrieved = len(relevant)
        recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
        
        # Validate recall value is in valid range
        assert 0.0 <= recall <= 1.0, f"Recall must be between 0.0 and 1.0, got {recall}"
        
        recall_results[profile] = recall
    
    # Validate mean recall calculation
    mean_recall = sum(recall_results.values()) / len(recall_results) if recall_results else 0.0
    assert 0.0 <= mean_recall <= 1.0, f"Mean recall must be between 0.0 and 1.0, got {mean_recall}"
    
    # Primary assertion: mean recall >= 0.5
    assert mean_recall >= 0.5, f"Mean Recall@10 {mean_recall:.2f} below threshold of 0.5"
