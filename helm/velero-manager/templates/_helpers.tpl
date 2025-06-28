{{/*
Expand the name of the chart.
*/}}
{{- define "velero-manager.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "velero-manager.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "velero-manager.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "velero-manager.labels" -}}
helm.sh/chart: {{ include "velero-manager.chart" . }}
{{ include "velero-manager.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "velero-manager.selectorLabels" -}}
app.kubernetes.io/name: {{ include "velero-manager.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "velero-manager.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "velero-manager.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the cluster role to use
*/}}
{{- define "velero-manager.clusterRoleName" -}}
{{- if .Values.rbac.clusterRole.name }}
{{- .Values.rbac.clusterRole.name }}
{{- else }}
{{- include "velero-manager.fullname" . }}
{{- end }}
{{- end }}

{{/*
Create the name of the cluster role binding to use
*/}}
{{- define "velero-manager.clusterRoleBindingName" -}}
{{- if .Values.rbac.clusterRoleBinding.name }}
{{- .Values.rbac.clusterRoleBinding.name }}
{{- else }}
{{- include "velero-manager.fullname" . }}
{{- end }}
{{- end }}