"""
Système de monitoring et métriques avancées pour le matching IA
- Monitoring en temps réel des performances
- Métriques de qualité des données
- Alertes automatiques
- Tableaux de bord intelligents
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import time

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Point de métrique avec timestamp"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = None

@dataclass
class Alert:
    """Alerte système"""
    id: str
    level: str  # "info", "warning", "error", "critical"
    message: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = None

class AIMonitoringService:
    """Service de monitoring IA en temps réel"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.alerts = []
        self.performance_baselines = {}
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Métriques à surveiller
        self.monitored_metrics = {
            "matching_accuracy": {"threshold": 0.8, "window": 100},
            "response_time": {"threshold": 2.0, "window": 50},
            "data_quality_score": {"threshold": 0.7, "window": 20},
            "anomaly_detection_rate": {"threshold": 0.1, "window": 30},
            "user_satisfaction": {"threshold": 0.75, "window": 50}
        }
    
    def start_monitoring(self):
        """Démarre le monitoring en arrière-plan"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("AI monitoring started")
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("AI monitoring stopped")
    
    def _monitoring_loop(self):
        """Boucle de monitoring en arrière-plan"""
        while self.is_monitoring:
            try:
                self._check_metrics()
                self._cleanup_old_data()
                time.sleep(30)  # Vérification toutes les 30 secondes
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Attendre plus longtemps en cas d'erreur
    
    def record_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None):
        """Enregistre une métrique"""
        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            metadata=metadata or {}
        )
        self.metrics_history[metric_name].append(metric_point)
        
        # Vérifier si une alerte doit être déclenchée
        self._check_metric_threshold(metric_name, value)
    
    def _check_metric_threshold(self, metric_name: str, value: float):
        """Vérifie si une métrique dépasse ses seuils"""
        if metric_name not in self.monitored_metrics:
            return
        
        config = self.monitored_metrics[metric_name]
        threshold = config["threshold"]
        
        # Déterminer le niveau d'alerte
        if value < threshold * 0.5:
            level = "critical"
        elif value < threshold * 0.7:
            level = "error"
        elif value < threshold * 0.9:
            level = "warning"
        else:
            return  # Pas d'alerte
        
        # Créer l'alerte
        alert = Alert(
            id=f"{metric_name}_{datetime.now().timestamp()}",
            level=level,
            message=f"{metric_name} is {value:.2f}, below threshold {threshold}",
            timestamp=datetime.now(),
            metadata={"metric_name": metric_name, "value": value, "threshold": threshold}
        )
        
        self.alerts.append(alert)
        logger.warning(f"Alert triggered: {alert.message}")
    
    def _check_metrics(self):
        """Vérifie les métriques en cours"""
        for metric_name, config in self.monitored_metrics.items():
            if metric_name in self.metrics_history:
                recent_values = list(self.metrics_history[metric_name])[-config["window"]:]
                if recent_values:
                    avg_value = np.mean([m.value for m in recent_values])
                    self._check_metric_threshold(metric_name, avg_value)
    
    def _cleanup_old_data(self):
        """Nettoie les anciennes données"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for metric_name in list(self.metrics_history.keys()):
            # Supprimer les métriques anciennes
            while (self.metrics_history[metric_name] and 
                   self.metrics_history[metric_name][0].timestamp < cutoff_time):
                self.metrics_history[metric_name].popleft()
        
        # Supprimer les alertes anciennes résolues
        self.alerts = [
            alert for alert in self.alerts 
            if not alert.resolved or alert.timestamp > cutoff_time
        ]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des métriques"""
        summary = {}
        
        for metric_name, history in self.metrics_history.items():
            if not history:
                continue
            
            values = [m.value for m in history]
            timestamps = [m.timestamp for m in history]
            
            summary[metric_name] = {
                "current_value": values[-1] if values else None,
                "average": np.mean(values) if values else 0,
                "min": np.min(values) if values else 0,
                "max": np.max(values) if values else 0,
                "trend": self._calculate_trend(values),
                "data_points": len(values),
                "last_updated": timestamps[-1].isoformat() if timestamps else None
            }
        
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcule la tendance d'une série de valeurs"""
        if len(values) < 2:
            return "stable"
        
        # Utiliser une régression linéaire simple
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculer la pente
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def get_alerts(self, level: Optional[str] = None, unresolved_only: bool = True) -> List[Dict[str, Any]]:
        """Retourne les alertes"""
        filtered_alerts = self.alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if unresolved_only:
            filtered_alerts = [a for a in filtered_alerts if not a.resolved]
        
        return [
            {
                "id": alert.id,
                "level": alert.level,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "metadata": alert.metadata
            }
            for alert in filtered_alerts
        ]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Marque une alerte comme résolue"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance détaillé"""
        summary = self.get_metrics_summary()
        alerts = self.get_alerts()
        
        # Calculer des scores de santé
        health_scores = {}
        for metric_name, data in summary.items():
            if metric_name in self.monitored_metrics:
                threshold = self.monitored_metrics[metric_name]["threshold"]
                current_value = data["current_value"]
                
                if current_value is not None:
                    health_score = min(1.0, current_value / threshold)
                    health_scores[metric_name] = health_score
        
        overall_health = np.mean(list(health_scores.values())) if health_scores else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health_score": overall_health,
            "health_status": self._get_health_status(overall_health),
            "metrics_summary": summary,
            "health_scores": health_scores,
            "active_alerts": len([a for a in alerts if not a["resolved"]]),
            "critical_alerts": len([a for a in alerts if a["level"] == "critical" and not a["resolved"]]),
            "recommendations": self._generate_recommendations(summary, alerts)
        }
    
    def _get_health_status(self, health_score: float) -> str:
        """Détermine le statut de santé global"""
        if health_score >= 0.9:
            return "excellent"
        elif health_score >= 0.7:
            return "good"
        elif health_score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _generate_recommendations(self, summary: Dict[str, Any], alerts: List[Dict[str, Any]]) -> List[str]:
        """Génère des recommandations basées sur les métriques et alertes"""
        recommendations = []
        
        # Recommandations basées sur les métriques
        for metric_name, data in summary.items():
            if data["trend"] == "decreasing":
                recommendations.append(f"Monitor {metric_name} closely - showing declining trend")
            
            if data["current_value"] is not None and metric_name in self.monitored_metrics:
                threshold = self.monitored_metrics[metric_name]["threshold"]
                if data["current_value"] < threshold * 0.8:
                    recommendations.append(f"Consider optimizing {metric_name} - currently below optimal range")
        
        # Recommandations basées sur les alertes
        critical_alerts = [a for a in alerts if a["level"] == "critical" and not a["resolved"]]
        if critical_alerts:
            recommendations.append("Address critical alerts immediately to prevent system degradation")
        
        warning_alerts = [a for a in alerts if a["level"] == "warning" and not a["resolved"]]
        if len(warning_alerts) > 5:
            recommendations.append("Multiple warning alerts detected - consider system review")
        
        return recommendations

# Instance globale du service de monitoring
monitoring_service = AIMonitoringService()

# Fonctions utilitaires pour l'intégration
def record_matching_accuracy(accuracy: float, candidate_id: str, offer_id: str):
    """Enregistre la précision d'un matching"""
    monitoring_service.record_metric(
        "matching_accuracy",
        accuracy,
        {"candidate_id": candidate_id, "offer_id": offer_id}
    )

def record_response_time(response_time: float, endpoint: str):
    """Enregistre le temps de réponse d'un endpoint"""
    monitoring_service.record_metric(
        "response_time",
        response_time,
        {"endpoint": endpoint}
    )

def record_data_quality_score(score: float, entity_type: str, entity_id: str):
    """Enregistre le score de qualité des données"""
    monitoring_service.record_metric(
        "data_quality_score",
        score,
        {"entity_type": entity_type, "entity_id": entity_id}
    )

def record_anomaly_detection(anomaly_rate: float, detection_type: str):
    """Enregistre le taux de détection d'anomalies"""
    monitoring_service.record_metric(
        "anomaly_detection_rate",
        anomaly_rate,
        {"detection_type": detection_type}
    )

def record_user_satisfaction(satisfaction: float, user_id: str, action: str):
    """Enregistre la satisfaction utilisateur"""
    monitoring_service.record_metric(
        "user_satisfaction",
        satisfaction,
        {"user_id": user_id, "action": action}
    )
