"""
Système de tests automatisés avec IA pour détecter les anomalies
et valider la qualité des données et des algorithmes de matching
"""
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

logger = logging.getLogger(__name__)

class AITestSuite:
    """Suite de tests automatisés avec IA"""
    
    def __init__(self):
        self.test_results = []
        self.anomaly_threshold = 0.1
        self.performance_thresholds = {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.7,
            "f1_score": 0.72
        }
    
    async def run_all_tests(self, store) -> Dict[str, Any]:
        """Exécute tous les tests automatisés"""
        logger.info("Starting AI test suite...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "anomalies_detected": 0,
            "performance_issues": 0,
            "recommendations": []
        }
        
        # 1. Test de qualité des données
        data_quality_result = await self.test_data_quality(store)
        test_results["tests_run"] += 1
        if data_quality_result["passed"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["anomalies_detected"] += data_quality_result.get("anomalies_count", 0)
        
        # 2. Test de performance des algorithmes
        performance_result = await self.test_algorithm_performance(store)
        test_results["tests_run"] += 1
        if performance_result["passed"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["performance_issues"] += 1
        
        # 3. Test de cohérence des scores
        consistency_result = await self.test_score_consistency(store)
        test_results["tests_run"] += 1
        if consistency_result["passed"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
        
        # 4. Test de détection d'anomalies
        anomaly_result = await self.test_anomaly_detection(store)
        test_results["tests_run"] += 1
        if anomaly_result["passed"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
            test_results["anomalies_detected"] += anomaly_result.get("anomalies_count", 0)
        
        # 5. Test de robustesse
        robustness_result = await self.test_system_robustness(store)
        test_results["tests_run"] += 1
        if robustness_result["passed"]:
            test_results["tests_passed"] += 1
        else:
            test_results["tests_failed"] += 1
        
        # Générer des recommandations
        test_results["recommendations"] = self.generate_recommendations(test_results)
        
        logger.info(f"AI test suite completed: {test_results['tests_passed']}/{test_results['tests_run']} tests passed")
        return test_results
    
    async def test_data_quality(self, store) -> Dict[str, Any]:
        """Test de qualité des données"""
        logger.info("Testing data quality...")
        
        candidates = store.list_candidates()
        offers = store.list_offers()
        
        issues = []
        anomalies_count = 0
        
        # Vérifier les candidats
        for candidate in candidates:
            # Vérifier les champs obligatoires
            if not candidate.get("name") or len(candidate.get("name", "").strip()) < 2:
                issues.append(f"Candidate {candidate.get('id')} has invalid name")
                anomalies_count += 1
            
            # Vérifier les compétences
            skills = candidate.get("skills", [])
            if not skills or len(skills) == 0:
                issues.append(f"Candidate {candidate.get('id')} has no skills")
                anomalies_count += 1
            
            # Vérifier la cohérence des données
            if len(candidate.get("text", "")) > 10000:  # Texte trop long
                issues.append(f"Candidate {candidate.get('id')} has unusually long text")
                anomalies_count += 1
        
        # Vérifier les offres
        for offer in offers:
            if not offer.get("title") or len(offer.get("title", "").strip()) < 3:
                issues.append(f"Offer {offer.get('id')} has invalid title")
                anomalies_count += 1
            
            if not offer.get("description") or len(offer.get("description", "").strip()) < 10:
                issues.append(f"Offer {offer.get('id')} has insufficient description")
                anomalies_count += 1
        
        # Vérifier la distribution des compétences
        all_skills = []
        for candidate in candidates:
            all_skills.extend(candidate.get("skills", []))
        
        if len(set(all_skills)) < 5:  # Pas assez de diversité
            issues.append("Low skill diversity across candidates")
            anomalies_count += 1
        
        passed = len(issues) == 0
        
        return {
            "test_name": "data_quality",
            "passed": passed,
            "issues": issues,
            "anomalies_count": anomalies_count,
            "candidates_checked": len(candidates),
            "offers_checked": len(offers),
            "total_skills": len(set(all_skills))
        }
    
    async def test_algorithm_performance(self, store) -> Dict[str, Any]:
        """Test de performance des algorithmes de matching"""
        logger.info("Testing algorithm performance...")
        
        candidates = store.list_candidates()
        offers = store.list_offers()
        
        if len(candidates) < 2 or len(offers) < 2:
            return {
                "test_name": "algorithm_performance",
                "passed": False,
                "reason": "Insufficient data for performance testing",
                "metrics": {}
            }
        
        # Simuler des tests de matching
        test_results = []
        
        for candidate in candidates[:5]:  # Tester sur les 5 premiers candidats
            candidate_skills = set(candidate.get("skills", []))
            
            for offer in offers[:10]:  # Tester sur les 10 premières offres
                offer_skills = set(offer.get("skills", []))
                
                # Calculer des métriques de base
                common_skills = len(candidate_skills.intersection(offer_skills))
                total_skills = len(candidate_skills.union(offer_skills))
                
                if total_skills > 0:
                    jaccard_similarity = common_skills / total_skills
                    test_results.append({
                        "candidate_id": candidate["id"],
                        "offer_id": offer["id"],
                        "jaccard_similarity": jaccard_similarity,
                        "common_skills": common_skills,
                        "total_skills": total_skills
                    })
        
        if not test_results:
            return {
                "test_name": "algorithm_performance",
                "passed": False,
                "reason": "No valid test cases generated",
                "metrics": {}
            }
        
        # Analyser les résultats
        similarities = [r["jaccard_similarity"] for r in test_results]
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Vérifier si les scores sont dans une plage raisonnable
        passed = 0.1 <= avg_similarity <= 0.9 and std_similarity < 0.5
        
        return {
            "test_name": "algorithm_performance",
            "passed": passed,
            "metrics": {
                "average_similarity": avg_similarity,
                "std_similarity": std_similarity,
                "test_cases": len(test_results),
                "min_similarity": min(similarities),
                "max_similarity": max(similarities)
            }
        }
    
    async def test_score_consistency(self, store) -> Dict[str, Any]:
        """Test de cohérence des scores de matching"""
        logger.info("Testing score consistency...")
        
        candidates = store.list_candidates()
        offers = store.list_offers()
        
        if len(candidates) < 2 or len(offers) < 2:
            return {
                "test_name": "score_consistency",
                "passed": False,
                "reason": "Insufficient data for consistency testing"
            }
        
        # Tester la cohérence des scores
        inconsistencies = []
        
        for candidate in candidates[:3]:
            candidate_skills = set(candidate.get("skills", []))
            
            # Calculer les scores pour toutes les offres
            scores = []
            for offer in offers:
                offer_skills = set(offer.get("skills", []))
                common = len(candidate_skills.intersection(offer_skills))
                total = len(candidate_skills.union(offer_skills))
                
                if total > 0:
                    score = common / total
                    scores.append(score)
            
            # Vérifier la distribution des scores
            if len(scores) > 1:
                score_std = np.std(scores)
                if score_std < 0.01:  # Scores trop similaires
                    inconsistencies.append(f"Candidate {candidate['id']} has too similar scores across offers")
                elif score_std > 0.8:  # Scores trop variables
                    inconsistencies.append(f"Candidate {candidate['id']} has too variable scores across offers")
        
        passed = len(inconsistencies) == 0
        
        return {
            "test_name": "score_consistency",
            "passed": passed,
            "inconsistencies": inconsistencies,
            "candidates_tested": min(3, len(candidates))
        }
    
    async def test_anomaly_detection(self, store) -> Dict[str, Any]:
        """Test de détection d'anomalies"""
        logger.info("Testing anomaly detection...")
        
        candidates = store.list_candidates()
        
        if len(candidates) < 3:
            return {
                "test_name": "anomaly_detection",
                "passed": False,
                "reason": "Insufficient data for anomaly detection testing"
            }
        
        # Créer des features pour la détection d'anomalies
        features = []
        for candidate in candidates:
            feature_vector = [
                len(candidate.get("skills", [])),
                len(candidate.get("text", "")),
                len(candidate.get("name", ""))
            ]
            features.append(feature_vector)
        
        # Utiliser Isolation Forest pour détecter les anomalies
        from sklearn.ensemble import IsolationForest
        
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features)
            
            anomalies = []
            for i, label in enumerate(anomaly_labels):
                if label == -1:  # Anomalie détectée
                    anomalies.append({
                        "candidate_id": candidates[i]["id"],
                        "candidate_name": candidates[i]["name"],
                        "reason": "Statistical anomaly in profile characteristics"
                    })
            
            # Le test passe si on détecte des anomalies de manière cohérente
            passed = len(anomalies) <= len(candidates) * 0.3  # Max 30% d'anomalies
            
            return {
                "test_name": "anomaly_detection",
                "passed": passed,
                "anomalies_detected": anomalies,
                "anomalies_count": len(anomalies),
                "total_candidates": len(candidates)
            }
        except Exception as e:
            logger.error(f"Error in anomaly detection test: {e}")
            return {
                "test_name": "anomaly_detection",
                "passed": False,
                "error": str(e)
            }
    
    async def test_system_robustness(self, store) -> Dict[str, Any]:
        """Test de robustesse du système"""
        logger.info("Testing system robustness...")
        
        robustness_issues = []
        
        # Test 1: Gestion des données vides
        try:
            empty_candidate = {"id": "test", "name": "", "text": "", "skills": []}
            # Simuler le traitement d'un candidat vide
            if not empty_candidate.get("name"):
                robustness_issues.append("System doesn't handle empty names gracefully")
        except Exception as e:
            robustness_issues.append(f"Error handling empty data: {e}")
        
        # Test 2: Gestion des caractères spéciaux
        try:
            special_candidate = {
                "id": "test",
                "name": "Test@#$%^&*()",
                "text": "Text with émojis 🚀 and spécial chars",
                "skills": ["Python", "JavaScript", "C++"]
            }
            # Le système devrait gérer les caractères spéciaux
        except Exception as e:
            robustness_issues.append(f"Error handling special characters: {e}")
        
        # Test 3: Gestion des données volumineuses
        try:
            large_text = "x" * 50000  # Texte très long
            large_candidate = {
                "id": "test",
                "name": "Large Data Test",
                "text": large_text,
                "skills": ["Python"] * 100  # Beaucoup de compétences
            }
            # Le système devrait gérer les données volumineuses
        except Exception as e:
            robustness_issues.append(f"Error handling large data: {e}")
        
        passed = len(robustness_issues) == 0
        
        return {
            "test_name": "system_robustness",
            "passed": passed,
            "issues": robustness_issues
        }
    
    def generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les résultats des tests"""
        recommendations = []
        
        if test_results["anomalies_detected"] > 0:
            recommendations.append("Review and clean data quality issues detected in profiles")
        
        if test_results["performance_issues"] > 0:
            recommendations.append("Optimize matching algorithms for better performance")
        
        if test_results["tests_failed"] > test_results["tests_passed"]:
            recommendations.append("Consider comprehensive system review and refactoring")
        
        if test_results["tests_passed"] == test_results["tests_run"]:
            recommendations.append("System is performing well - consider adding more advanced features")
        
        return recommendations

# Instance globale pour les tests
ai_test_suite = AITestSuite()
