<!-- MKU Project Paper | CrisisMap | Kinga Hinzano | A4 | Times New Roman 12pt | 1.5 spacing -->

<div align="center">

# CHAPTER THREE: METHODOLOGY

</div>

## 3.1 Introduction
The methodology chapter described the structured approach and specific techniques used to conduct the research and develop the global CrisisMap platform. It detailed the fact-finding methods used to gather requirements from international stakeholders, the tools utilized for global data analysis, and the development framework that guided the implementation process. The goal of this chapter was to provide a transparent and reproducible roadmap that justified the technical choices made for a multi-regional analytical system.

## 3.2 Research Design
The student employed an exploratory research design combined with an applied development approach. The exploratory phase was necessary to understand the complexities of global conflict data and the specific operational needs of the International Crisis Monitoring Network (ICMN). This involved a qualitative assessment of the organization’s current manual CSV workflows across multiple continents and a quantitative analysis of historical international conflict datasets. The applied development phase focused on the iterative construction of the system components, utilizing the findings from the research phase to inform the system's global architecture.

The study utilized an Agile development methodology, specifically the Scrum framework, which allowed for rapid prototyping and continuous feedback from international analysts. This choice was justified by the volatile nature of global conflict data, as the technical landscape of international APIs and data sources was subject to frequent changes. By working in iterative sprints, the student was able to deliver functional components—such as the multi-regional CSV ingestion engine—early in the project lifecycle, ensuring that any technical risks were identified promptly.

## 3.3 Data Collection Techniques
The gathering of facts and data was a multi-modal process that ensured a comprehensive understanding of the global problem domain. The following techniques were used:

### 3.3.1 Interviews
Structured and semi-structured interviews were conducted with key stakeholders at the ICMN international headquarters, including global data analysts and regional coordinators. These interviews were instrumental in uncovering the hidden inefficiencies of managing disparate CSV formats from Africa, Asia, and Europe. The student asked specific questions regarding the time taken to harmonize continental datasets and the specific metrics that international analysts felt were missing from their current regional tools. The insights gained from these sessions formed the basis of the system's functional requirements.

### 3.3.2 Observation
The student performed direct observation of the global data consolidation process at the ICMN office. By watching how CSV files from different continents were received, cleaned, and merged, the student identified significant bottlenecks in the "global data lifecycle." It was observed that the manual mapping of columns from different regional schemas was the primary cause of reporting delays. This observation led to the decision to implement an adaptive column mapping engine that automated the standardization of multi-regional CSVs.

### 3.3.3 Document Review
A thorough review of existing international situational reports and multi-regional spreadsheets was conducted. The student analyzed the data structure of various partner organizations in Asia and Europe. This review revealed that while the ICMN received a vast amount of data, the lack of standardized schemas prevented cross-continental trend analysis. This finding justified the development of the "Adaptive Ingestion" module within CrisisMap to support global interoperability.
...
