<!-- MKU Project Paper | CrisisMap | Kinga Hinzano | A4 | Times New Roman 12pt | 1.5 spacing -->

<div align="center">

# APPENDIX

</div>

## A. Global Organisational Structure Diagram
The organizational structure of the International Crisis Monitoring Network (ICMN) was modeled to show the reporting lines from regional offices across the world.

```mermaid
graph TD
    A[Global Headquarters] --> B[Africa Regional Office]
    A --> C[Asia Regional Office]
    A --> D[Europe Regional Office]
    B & C & D -- CSV Datasets --> E[CrisisMap Global Backend]
    E -- Predictive Insights --> F[International Response Teams]
```
*Figure A.1: Global Organisational Structure Diagram*

...
## D. Global Project Gantt Chart
```mermaid
gantt
    title CrisisMap Global Development Schedule
    dateFormat  YYYY-MM-DD
    section Planning
    Initial Research & Global Proposal :a1, 2024-01-01, 14d
    section Analysis
    International Stakeholder Interviews :a2, after a1, 21d
    Multi-Regional Requirement Spec :a3, after a2, 14d
    section Design
    Global Architecture Design :a4, after a3, 21d
    Multi-Regional Database Modeling :a5, after a4, 14d
    section Implementation
    Adaptive CSV Engine Development :a6, after a5, 42d
    Global 3D Dashboard Design :a7, after a6, 28d
    section Testing
    Multi-Continental Data Validation :a8, after a7, 21d
```
...
