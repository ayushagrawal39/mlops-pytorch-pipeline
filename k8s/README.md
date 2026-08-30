# Kubernetes apply order

Images must exist in the cluster (Minikube: `eval $(minikube docker-env)` then `docker build`).

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/training-job.yaml
kubectl wait --for=condition=complete job/model-training -n ml-training --timeout=30m
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training
kubectl port-forward svc/model-serving 8080:80 -n ml-training
```

PVCs use `ReadWriteOnce` so they work on default Minikube storage.
