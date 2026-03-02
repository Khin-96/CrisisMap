# Crisis Map Proposal

## Chapter 1: Introduction

### 1.1 Introduction
CrisisMap is proposed as a data-driven early warning and decision-support platform for conflict and humanitarian risk monitoring in the Great Lakes Region, with a practical implementation focus on Eastern Democratic Republic of the Congo (DRC). The system is designed to convert fragmented event and displacement records into actionable intelligence for humanitarian teams, peacebuilding actors, analysts, and policy institutions. 

The proposal responds to a persistent operational challenge: crises escalate faster than many response systems can analyse and communicate risk. By combining near-real-time conflict data, displacement information, and predictive analytics, CrisisMap seeks to shorten the interval between signal detection and informed intervention.

### 1.2 Background of the Study
Forced displacement and conflict volatility remain defining features of the current global risk landscape. UNHCR reported that **123.2 million** people were forcibly displaced worldwide at the end of 2024 (UNHCR, 2025a), while IDMC reported **83.4 million** people internally displaced at the same point, with **73.5 million** linked to conflict and violence (IDMC, 2025). These patterns indicate persistent fragility and the need for anticipatory, data-enabled response models.

In the DRC context, the crisis has remained complex and regionally destabilizing. A UNHCR regional update dated 1 March 2025 notes a **USD 2.54 billion** Humanitarian Response Plan targeting **11 million** people, including **7.8 million internally displaced people** (UNHCR, 2025b). This magnitude confirms that conventional periodic reporting alone is insufficient for rapid operational decisions.

Concurrently, digital infrastructure has expanded globally, enabling stronger technical feasibility for crisis informatics. ITU estimates that 6 billion people were online in 2025, although major inequality persists in low-income contexts (ITU, 2025). This uneven but growing digital base supports the case for web-based, low-bandwidth-conscious tools tailored to mixed-connectivity environments.

### 1.3 Problem Statement
Current conflict and humanitarian information ecosystems are affected by five recurrent constraints:

1. **Data fragmentation**: critical indicators are distributed across ACLED, HDX/OCHA, IOM DTM, and local records with inconsistent schemas.
2. **Latency**: manual compilation and static dashboards delay recognition of fast-changing risk patterns.
3. **Limited predictive capability**: many systems remain descriptive, not anticipatory.
4. **Weak cross-source triangulation**: operational users struggle to compare conflict events with displacement, climate, or socio-economic stressors in one workflow.
5. **Decision bottlenecks**: analysts spend excessive time cleaning and merging data rather than generating policy-ready insight.

The practical effect is delayed early warning, suboptimal targeting of interventions, and increased humanitarian cost.

### 1.4 Problem Solution
CrisisMap proposes an integrated architecture that unifies ingestion, analytics, and communication in one operational platform:

1. **Unified data pipeline** to ingest conflict and humanitarian indicators from ACLED, HDX-compatible sources, and structured local uploads.
2. **Persistent local intelligence layer** using SQLite for resilient, offline-capable analytical work.
3. **Risk analytics suite** including trend analysis, anomaly detection, and short-horizon forecasting.
4. **Interactive geospatial interface** for hotspot detection, event density mapping, and temporal comparison.
5. **Decision outputs** such as alerts, risk brief templates, and exportable summaries for coordination forums.

This solution aligns with the broader shift from reactive reporting to anticipatory action and risk-informed planning (WMO, 2025a; UNDRR, 2025).

### 1.5 Objectives of the System
The proposed system aims to:

1. Consolidate multi-source crisis data into a standardized, queryable repository.
2. Provide near-real-time visualization of conflict incidents and displacement patterns.
3. Implement machine-learning-assisted forecasts for short-term risk evolution.
4. Detect statistical and spatial anomalies to surface early warning signals.
5. Support evidence-based planning through automated analytical summaries.
6. Improve the speed, consistency, and transparency of crisis intelligence workflows.

### 1.6 Research Questions
The project is guided by the following research questions:

1. How effectively can multi-source conflict and humanitarian datasets be integrated into one operational model without unacceptable data-loss or bias?
2. Which predictive and anomaly-detection techniques provide the most reliable short-horizon warning signals in volatile conflict environments?
3. To what extent does an integrated dashboard improve analyst response time and decision quality compared with manual workflows?
4. What governance controls are required to ensure responsible use of sensitive crisis data?

### 1.7 Scope of the Study
**Geographical scope:** Great Lakes Region, with primary operational focus on Eastern DRC.

**Thematic scope:** conflict events, fatalities, displacement indicators, hotspot migration, and selected contextual drivers.

**Technical scope:** data ingestion APIs, database persistence, analytics models, and user-facing dashboards.

**Time scope:** phased implementation over six months (April 2026 to September 2026), followed by pilot evaluation.

**Out of scope (Phase 1):** autonomous decision-making, classified data ingestion, and full multilingual NLP pipelines.

### 1.8 Project Description
CrisisMap is a modular platform implemented with a FastAPI backend and Streamlit frontend. It supports both live data synchronization and historical imports.

**Core modules:**
1. **Ingestion layer**: ACLED API pull, HDX-compatible endpoints, CSV ingestion.
2. **Storage layer**: normalized relational schema in SQLite.
3. **Analytics layer**:
   - Time-series trend decomposition
   - Short-horizon forecasting (e.g., Random Forest, Gradient Boosting)
   - Anomaly detection (statistical and spatial)
   - Actor-pattern exploration
4. **Presentation layer**:
   - Live monitor
   - Geospatial intelligence view
   - Forecast and risk panel
   - Export centre

**Expected users:** humanitarian coordination teams, policy analysts, peace and security researchers, and regional operational planners.

### 1.9 System Requirements

#### Functional Requirements
1. Pull and update conflict/displacement datasets from configured sources.
2. Validate, transform, and persist records with metadata provenance.
3. Compute risk scores and short-horizon predictions.
4. Generate map-based and time-series visualizations.
5. Trigger alert flags for anomalous spikes in events/fatalities/displacement.
6. Export filtered datasets and summary reports (CSV/PDF-ready structures).

#### Non-Functional Requirements
1. **Performance**: dashboard response under 2 seconds for common filtered views.
2. **Availability**: 99% uptime target during pilot operations.
3. **Scalability**: modular service design to support migration from SQLite to managed SQL backends.
4. **Security**: role-based access controls and secure secret handling for API credentials.
5. **Interoperability**: standards-based API interactions and structured metadata.
6. **Auditability**: ingestion logs, transformation logs, and model version tracking.

#### Software and Hardware Requirements
1. Python 3.10+ runtime.
2. FastAPI, Streamlit, Pandas, NumPy, scikit-learn, Plotly, Folium, SQLAlchemy.
3. Minimum 8 GB RAM for local analyst workstation.
4. Stable internet for live sync; local mode for offline analysis.

### 1.10 Risk and Mitigation

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| Data access interruptions from external APIs | Medium | High | Caching, retry policies, and fallback local datasets |
| Inconsistent data schemas across providers | High | High | Schema mapping layer with strict validation and transformation tests |
| Model drift in rapidly changing conflict conditions | Medium | High | Scheduled retraining, rolling validation, and performance monitoring |
| Misinterpretation of predictive outputs | Medium | High | Confidence intervals, uncertainty notes, and analyst guidance prompts |
| Cybersecurity/privacy exposure | Medium | High | Credential vaulting, access control, audit logs, and least-privilege design |
| Funding constraints for long-term maintenance | Medium | Medium | Phased rollout, open-source stack, and cost-controlled hosting |
| Connectivity limitations in field settings | High | Medium | Offline-capable cache, low-bandwidth visual modes, periodic sync |

### 1.11 Budget
Estimated Phase 1 budget (6 months, pilot-grade deployment):

| Budget Item | Cost (USD) |
|---|---:|
| Software engineering and integration | 15,000 |
| Data engineering and QA | 8,000 |
| ML modeling and validation | 6,500 |
| UX/dashboard development | 5,500 |
| Cloud/hosting, backup, and monitoring | 3,800 |
| Security hardening and access controls | 2,500 |
| Documentation and user training | 2,200 |
| Pilot operations and stakeholder workshops | 3,500 |
| Contingency (10%) | 4,700 |
| **Total Estimated Budget** | **51,700** |

### 1.12 Project Timeline (Gantt Chart)
Implementation window: **April 2026 to September 2026**.

| Work Package | Apr | May | Jun | Jul | Aug | Sep |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Requirements validation and governance framework | X | X |  |  |  |  |
| Data ingestion connectors and schema mapping | X | X | X |  |  |  |
| Database and API services hardening |  | X | X | X |  |  |
| Analytics model development and back-testing |  |  | X | X | X |  |
| Dashboard integration and user acceptance testing |  |  |  | X | X |  |
| Pilot deployment and training |  |  |  |  | X | X |
| Monitoring, evaluation, and close-out report |  |  |  |  |  | X |

### 1.13 Conclusion
The proposed CrisisMap system addresses a clearly documented operational gap between crisis data availability and timely decision intelligence. Given current global displacement pressures, escalating multi-hazard risk, and the push for universal early-warning coverage by 2027, an integrated and analytically robust platform is both relevant and urgent. This proposal outlines a feasible, phased, and governance-aware path for implementation.

## Chapter 2: Literature Review

### 2.0 Introduction
This chapter reviews prior scholarship and institutional practice on crisis mapping, conflict event monitoring, displacement analytics, and early warning systems. It examines what is already established, where consensus exists, and which methodological weaknesses remain.

### 2.1 Past Information (What has been said before about my topic)
Conflict and humanitarian analysis has progressively shifted from static reporting to dynamic, geospatial, and near-real-time monitoring. ACLED is widely used for event-level conflict data and publishes structured methodology and API guidance for granular temporal and actor-based analysis (ACLED, 2025a; ACLED, 2025b).

At global scale, institutional evidence confirms growing pressure on risk systems. UNDRR (2025) highlights escalating macroeconomic disaster losses and argues for stronger risk-informed investment; WMO (2025a) and WMO (2025b) situate multi-hazard early warning systems as a global priority, with measurable but uneven progress. 

Displacement literature similarly emphasizes persistent scale and protracted vulnerability. IDMC (2025) and UNHCR (2025a) both report record-level displacement burdens, while UNHCR’s Eastern DRC update shows how regional conflict translates directly into operational humanitarian financing and protection requirements (UNHCR, 2025b).

Digital inclusion evidence adds an implementation constraint: while internet uptake is rising, structural access inequality remains severe, especially in low-income settings (ITU, 2025). Therefore, crisis platforms must combine modern analytics with connectivity-aware design.

### 2.2 Additions of Knowledge (Who said what)
The literature contributes across complementary dimensions:

1. **Data generation and coding**: ACLED provides high-frequency event coding frameworks and access modalities that enable localized violence trend analysis (ACLED, 2025b).
2. **Scale diagnostics**: IDMC and UNHCR quantify the magnitude and evolution of displacement, establishing the empirical urgency for anticipatory systems (IDMC, 2025; UNHCR, 2025a).
3. **Early warning architecture**: WMO’s EW4All and global status reporting define a four-pillar, end-to-end model for risk knowledge, detection, communication, and preparedness (WMO, 2025a; WMO, 2025b).
4. **Risk-economy linkage**: UNDRR frames risk reduction as a financial stability and development necessity rather than a narrow humanitarian issue (UNDRR, 2025).
5. **Platform interoperability and API practice**: CKAN documentation and HDX HAPI discussions indicate movement toward machine-readable humanitarian interoperability, though maturity varies by source and indicator class (CKAN, 2025; OCHA Centre for Humanitarian Data, 2024).

CrisisMap adds value by operationally combining these strands into one reproducible workflow: event ingestion, displacement correlation, predictive analytics, and policy-facing visualization.

### 2.3 Criticism & Comparisons
Despite important progress, existing approaches reveal persistent limitations:

1. **Descriptive dominance**: many systems explain what has happened, but not what is likely to happen next.
2. **Interoperability gaps**: API availability does not guarantee harmonized semantics across datasets.
3. **Validation asymmetry**: predictive claims are often under-documented in terms of drift, confidence calibration, and external validity.
4. **Usability tension**: technical dashboards may overwhelm non-technical decision-makers.
5. **Ethical and governance concerns**: use of sensitive geospatial conflict data can expose vulnerable groups if safeguards are weak.

Compared with single-source dashboards, CrisisMap’s proposed architecture is explicitly multi-source and analytics-enabled. Compared with purely model-driven approaches, it preserves analyst oversight and uncertainty communication. Compared with static humanitarian reporting, it supports rolling updates and faster iteration cycles.

The central criticism to anticipate is that predictive tools can create false precision in volatile contexts. The design response in this proposal is methodological transparency: explicit confidence reporting, human-in-the-loop interpretation, and periodic model audit.

## Chapter 3: Research Methodology

### 3.1 Introduction
This chapter defines the methodological framework for the CrisisMap project and explains how the system will be developed, validated, and evaluated in real operational settings. The methodology is structured to ensure scientific rigor, practical relevance, and reproducibility across the full project lifecycle. Specifically, it details the research design, target population, sampling strategy, data collection methods, analytical procedures, data presentation formats, and ethical safeguards. The chapter is intended to provide a defensible empirical foundation for both platform engineering and evidence-based performance assessment.

### 3.2 Research Design
The study adopts a **mixed-methods, design-science-oriented research design**. This combination is appropriate because CrisisMap is both a technical artifact and a decision-support intervention that must be evaluated quantitatively and qualitatively.

1. **Design Science Research (DSR) orientation**: The project follows design science principles by iteratively building and evaluating an artifact that addresses a clearly defined socio-technical problem (Hevner et al., 2004; Peffers et al., 2007).
2. **Quantitative strand**: Quantitative procedures assess model and system performance using measurable indicators.
3. **Qualitative strand**: Qualitative inquiry captures analyst interpretation, usability concerns, and operational trust dynamics that are not fully measurable through numeric indicators alone.
4. **Integration logic**: Findings from both strands are triangulated to determine whether statistical performance translates into practical decision utility (Johnson et al., 2007).

#### Quantitative Evaluation Dimensions
1. Forecast error metrics for short-horizon predictions (e.g., MAE, RMSE).
2. Event hotspot detection quality (precision, recall, false-alarm rate).
3. Data pipeline reliability (ingestion success rate, synchronization latency).
4. Dashboard performance (response time under typical query loads).

#### Qualitative Evaluation Dimensions
1. Perceived usefulness for early warning and planning.
2. Interpretability of model outputs and uncertainty indicators.
3. Workflow fit across humanitarian and policy-analysis contexts.
4. Barriers to adoption, including bandwidth, training, and governance concerns.

### 3.3 Target Population
The primary population includes institutional and professional users who produce or consume crisis intelligence in the Great Lakes Region, especially Eastern DRC.

1. Humanitarian analysts and information management officers.
2. Peace and security analysts in regional and international organizations.
3. Policy researchers and planning officers in government-linked and multilateral structures.
4. Operational coordination actors using conflict and displacement data for prioritization.

Secondary participants include data engineers, system administrators, and subject-matter experts involved in validation workflows. The population is intentionally heterogeneous because system effectiveness depends on both technical performance and cross-functional interpretability.

### 3.4 Sampling Design

#### 3.4.1 Sampling Techniques
A multi-stage non-probability-plus-probability strategy will be used to balance feasibility and representativeness:

1. **Purposive sampling (Stage 1)**: Select organizations and operational units that routinely engage with conflict and displacement intelligence.
2. **Cluster sampling (Stage 2)**: Group participants by organizational type (humanitarian, policy, research, and coordination).
3. **Stratified random sampling (Stage 3)**: Within each cluster, sample participants by role seniority and technical literacy to reduce selection bias.

#### 3.4.2 Sample Size
1. **Quantitative component**: A minimum of 150-250 structured observations (dashboard sessions, alert evaluations, or validated prediction windows), sufficient for inferential analysis under pilot constraints.
2. **Qualitative component**: Approximately 20-30 interviews and 3-5 focus groups, with final size determined by thematic saturation.
3. **System logs**: Continuous collection of anonymized interaction and performance records during the pilot window.

### 3.5 Data Collection Methods
Data collection will combine instrument-based, system-generated, and field-informed methods:

1. **Structured surveys**: Capture user assessments of utility, clarity, and trust.
2. **Semi-structured interviews**: Explore interpretive judgments and institutional constraints in depth.
3. **Focus group discussions (FGDs)**: Elicit comparative perspectives and shared operational priorities.
4. **Direct observation**: Observe live analytical workflows and decision meetings where feasible.
5. **System telemetry and usage analytics**: Collect anonymized logs (query frequency, filter behavior, dashboard latency, alert review actions).
6. **Archival validation datasets**: Compare predictions and alerts against verified event/displacement records from authoritative sources.

Data collection instruments will be piloted and refined before full deployment to improve clarity and measurement validity.

### 3.6 Data Analysis
The analysis plan is explicitly mixed-methods and sequentially integrated.

#### Quantitative Analysis
1. Descriptive statistics to profile usage patterns and baseline operational metrics.
2. Inferential testing (e.g., t-tests, chi-square tests, regression models) to assess whether observed differences are statistically meaningful.
3. Time-series and model-evaluation diagnostics for predictive modules.
4. Error decomposition to identify systematic underperformance by region, actor type, or event class.

#### Qualitative Analysis
1. Thematic analysis using a structured coding framework to identify recurrent patterns in interviews and FGDs (Braun & Clarke, 2006).
2. Cross-case comparison across organization types to isolate context-specific versus generalizable insights.
3. Validation of emerging interpretations through peer debriefing and codebook revision.

#### Integration of Findings
Quantitative and qualitative results will be merged during interpretation to produce convergent, complementary, or divergent evidence profiles. This integration supports stronger conclusions about both technical performance and real-world usability.

### 3.7 Data Presentation Methods
Findings will be presented in formats aligned to technical and policy audiences:

1. **Quantitative outputs**: tables, confidence intervals, trend lines, heat maps, and model performance dashboards.
2. **Qualitative outputs**: thematic matrices, coded excerpts, and narrative synthesis of operational themes.
3. **Integrated outputs**: decision briefs combining statistical evidence, uncertainty statements, and practitioner interpretation.

All presentation outputs will clearly distinguish observed evidence from inferred conclusions to preserve analytical transparency.

### 3.8 Conclusion
This chapter has established a rigorous and operationally grounded methodology for CrisisMap. By combining design science with mixed-methods evaluation, the study can test whether the platform is not only technically accurate but also institutionally usable and ethically responsible. The methodology provides a coherent roadmap for implementation, validation, and iterative improvement, thereby strengthening the credibility and utility of the final system.
## References (APA 7th Edition)

ACLED. (2025a). *Conflict data*. [https://acleddata.com/conflict-data](https://acleddata.com/conflict-data)

ACLED. (2025b). *API documentation*. [https://acleddata.com/acled-api-documentation](https://acleddata.com/acled-api-documentation)

Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. *Qualitative Research in Psychology, 3*(2), 77-101. [https://doi.org/10.1191/1478088706qp063oa](https://doi.org/10.1191/1478088706qp063oa)

CKAN. (2025). *API guide (Version 2.11.4 documentation)*. [https://docs.ckan.org/en/2.11/api/](https://docs.ckan.org/en/2.11/api/)

Hevner, A. R., March, S. T., Park, J., & Ram, S. (2004). Design science in information systems research. *MIS Quarterly, 28*(1), 75-105. [https://doi.org/10.2307/25148625](https://doi.org/10.2307/25148625)

Internal Displacement Monitoring Centre. (2025). *Global report on internal displacement 2025*. [https://www.internal-displacement.org/global-report/](https://www.internal-displacement.org/global-report/)

International Telecommunication Union. (2025, November 17). *Global number of Internet users increases, but disparities deepen key digital divides (Facts and Figures 2025 press release)*. [https://www.itu.int/en/mediacentre/Pages/PR-2025-11-17-Facts-and-Figures.aspx](https://www.itu.int/en/mediacentre/Pages/PR-2025-11-17-Facts-and-Figures.aspx)

Johnson, R. B., Onwuegbuzie, A. J., & Turner, L. A. (2007). Toward a definition of mixed methods research. *Journal of Mixed Methods Research, 1*(2), 112-133. [https://doi.org/10.1177/1558689806298224](https://doi.org/10.1177/1558689806298224)

OCHA Centre for Humanitarian Data. (2024, June 27). *Announcing the HDX Humanitarian API*. [https://centre.humdata.org/announcing-the-hdx-humanitarian-api/](https://centre.humdata.org/announcing-the-hdx-humanitarian-api/)

Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. In T. Pries-Heje, V. K. Patel, & D. Bunkowski (Eds.), *Advances in theory and practice of emerging markets* (pp. 45-77). Springer. [https://doi.org/10.1007/978-3-540-76731-0_14](https://doi.org/10.1007/978-3-540-76731-0_14)

UNHCR. (2025a, June 12). *Global trends*. [https://www.unhcr.org/what-we-do/reports-and-publications/global-trends](https://www.unhcr.org/what-we-do/reports-and-publications/global-trends)

UNHCR. (2025b, March 1). *Eastern DRC situation - Regional external update #5 - 28 February 2025*. [https://data.unhcr.org/en/documents/details/114790](https://data.unhcr.org/en/documents/details/114790)

United Nations Office for Disaster Risk Reduction. (2025). *Global Assessment Report 2025: Resilience pays*. [https://www.undrr.org/gar2025](https://www.undrr.org/gar2025)

World Meteorological Organization. (2025a). *Early Warnings for All (EW4All)*. [https://wmo.int/site/knowledge-hub/programmes-and-initiatives/early-warnings-all-ew4all](https://wmo.int/site/knowledge-hub/programmes-and-initiatives/early-warnings-all-ew4all)

World Meteorological Organization. (2025b, November 12). *Global status of multi-hazard early warning systems 2025*. [https://wmo.int/publication-series/global-status-of-multi-hazard-early-warning-systems-2025](https://wmo.int/publication-series/global-status-of-multi-hazard-early-warning-systems-2025)

