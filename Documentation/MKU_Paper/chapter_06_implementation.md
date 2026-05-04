<!-- MKU Project Paper | CrisisMap | Kinga Hinzano | A4 | Times New Roman 12pt | 1.5 spacing -->

<div align="center">

# CHAPTER SIX: IMPLEMENTATION

</div>

## 6.1 Introduction
The implementation phase involved the actual construction of the global CrisisMap platform based on the designs established in the previous chapter. This phase encompassed the selection of development tools capable of handling multi-regional data, the coding of the system components, and the execution of a comprehensive testing plan. The student adopted a modular coding approach, ensuring that each part of the system—from the adaptive CSV adapters to the global ML pipeline—was thoroughly tested. This chapter described the technical environment used for global-scale development and the results of the validation process.

## 6.2 Development Environment and Tools
The development of CrisisMap was carried out in a high-performance environment to support the processing of massive multi-regional datasets. The following tools were utilized:

### 6.2.1 Coding Tools
The primary integrated development environment (IDE) was **Visual Studio Code**, chosen for its support for global development standards. **Git** was used for version control. The backend was implemented in **Python 3.12**, utilizing the FastAPI framework to support concurrent requests from international users. The frontend was developed using **Next.js 14**, providing a high-performance interactive 3D globe for global monitoring.

### 6.2.2 Testing Tools
For automated testing of the multi-regional logic, the student utilized the **Pytest** framework. **Postman** was used for validating API endpoints across different simulated regional environments. For the frontend, **Jest** ensured that the global 3D visualizations rendered correctly on various devices.

## 6.3 System Test Plan and Results
A rigorous test plan was executed to ensure that the system met the functional requirements for a global platform.

### 6.3.1 Unit and Integration Testing
Unit tests focused on individual functions, such as the `CSVAdapter._read_file_robust()` method, which was designed to handle various international CSV encodings. Integration testing verified the replication of data from local storage to the global MongoDB Atlas instances. The results of these tests were positive, with a 98% pass rate across the multi-regional test cases.

### 6.3.2 System and User Acceptance Testing
System testing involved running the complete platform using historical datasets from Africa, Asia, and Europe. The student validated the accuracy of the predictive models across different continents. User Acceptance Testing (UAT) was conducted with a small group of global analysts from the ICMN. They were asked to perform specific tasks, such as uploading a multi-regional CSV file and identifying global conflict trends on the 3D map. The feedback was used to refine the international user experience and the clarity of the global alert system.

Table 6.1: Unit Testing Results
| Module | Test Case | Expected Result | Actual Result | Status |
|---|---|---|---|---|
| Ingestion | Global CSV Mapping | Auto-detect columns | Schema identified | Pass |
| Database | Atlas Replication | Sync with MongoDB | Data replicated | Pass |
| ML Pipeline | Global Inference | Generate forecast | Multi-regional JSON | Pass |
| WebSocket | International Alert | Broadcast to all clients | Global sync | Pass |
...
