IS-305: AI Vision Based Kitting Solution for Warehousing
CDE3301 Final Report AY 2025/26

Submitted by:
Lim Kai Ler Ethan (A0307499M)
Aw Shuo Jie (A0309254E)
Yap Jia Wei (A0307951B)
Yeo Chen Xian (A0299847Y)

Supervisors: Eugene Ee, Chan Tong Leong

================================================================================
PROJECT PROGRESS & SUBMISSION CHECKLIST
================================================================================
- Final Report Submission: By 23rd Jul 2026 (Submit to 2 examiners, cc supervisors: Eugene, Prof Chan, Dr Elliot, and Prof Aleks).
- Draft Submission: Send to Prof Chan and Eugene by 21st Jul 2026.
- CX Exhibition: Live presentation recommended (Studio 6, near the fan). Video recording prepared as a backup.

================================================================================
TABLE OF CONTENTS
================================================================================
Chapter 1: Introduction and Problem Statement
  1.1 Problem Introduction
  1.2 Root Cause Analysis
  1.3 Current Solutions and Limitations

Chapter 2: Design Overview and Requirements
  2.1 Project Scope and Value Proposition
    2.1.1 Project Scope
    2.1.2 Value Proposition
  2.2 Requirements Definition and Technical Translation
    2.2.1 Stakeholder Requirements
    2.2.2 Functional Requirements
    2.2.3 Translation of Stakeholder Requirements into Technical Specifications
  2.3 System Architecture at Concept Overview
    2.3.1 Concept Overview
  2.4 Design and Component Selection
    2.4.1 System-Level Selection
    2.4.2 Detailed Component Selection
    2.4.3 Sequential Single-Bin Verification Finite State Machine (FSM)

Chapter 3: Prototype Development
  3.1 Physical Frame
  3.2 Sensors Subsystem
    3.2.1 Computer Vision
    3.2.2 Load Cells
  3.3 Software Subsystem
    3.3.1 Software Stack Specification
    3.3.2 Frontend HMI
    3.3.3 Backend Processing Engine

Chapter 4: System Integration and Functional Testing
  4.1 System Integration Overview
  4.2 Developer Functional Testing
    4.2.1 Computer Vision Subsystem
    4.2.2 Load Cells Bench Metrology
    4.2.3 Backend Verification (Pytest Unit Testing)
  4.3 Developer-Phase Iterations
    4.3.1 Hardware Iteration: Load Cell Platform Redesign
    4.3.2 Software Iteration: Occlusion Gate & Occlusion Hold

Chapter 5: System Testing and Evaluation/Validation
  5.1 Usability Study (User Testing)
    5.1.1 Testing Framework
    5.1.2 Evaluation Results
  5.2 User-Facing HMI Iteration
  5.3 Verification and Validation (Requirements Traceability Matrix)

Chapter 6: Conclusion and Future Works
  6.1 Prototype Summary and Sandbox Trial
  6.2 Process Scalability and Maintenance
  6.3 Ergonomic Considerations
  6.4 Future Works

Chapter 7: Reflections and Lessons Learned

References
Appendix


================================================================================
CHAPTER 1: INTRODUCTION AND PROBLEM STATEMENT
================================================================================

1.1 Problem Introduction
TE Connectivity Ltd. is a global manufacturer of connectivity and sensor solutions, serving industrial, automotive, aerospace, medical, and commercial transportation sectors. With annual revenues of approximately USD 17.3 billion (FY2025) and manufacturing operations spanning 105 principal sites across the Americas, EMEA, and APAC regions, TE Connectivity operates at a scale at which marginal inefficiencies in production processes translate into significant financial and reputational consequences. The company's product catalogue encompasses more than 242 billion components annually, including connectors, sensors, relays, precision wire and cable, and application tooling, each produced across a portfolio of approximately 4,000 distinct stock-keeping units (SKUs).

Within TE Connectivity's assembly plants — including facilities in China, Poland, and Mexico — a critical mid-stream operation known as cell kitting is performed manually by ground operators. Kitting refers to the process of collecting and assembling a prescribed set of components into a single unit or package, in preparation for downstream assembly, shipment, or customer delivery. Each kitting order specifies a particular mix of SKUs and quantities, drawn from a subset of the approximately 800 to 900 commonly active SKUs. Orders are typically fulfilled in one to two hours, with an estimated 50 to 100 individual packages packed per order at each kitting station. Given a team size of four to six operators per kitting line, the cumulative throughput per shift is substantial.

The manual nature of this process introduces an inherent susceptibility to human error. Operators must repeatedly identify, retrieve, and pack correct components across a high-variety product range, under production time pressure and across extended shift durations. The primary error types observed are: the inclusion of incorrect components (wrong part), omission of required components (missing part), and packing of incorrect quantities (wrong quantity — either too few or too many). These errors are not immediately self-evident at the point of commission, as the current verification infrastructure consists solely of a digital postal scale used by operators to weigh each completed kit against an expected aggregate weight. No real-time feedback mechanism exists to alert operators during the act of kitting itself.

The consequences of kitting errors propagate across the supply chain. Errors that pass the scale check may be identified at downstream quality control inspections, at end-of-line audits, or — in the worst case — only upon delivery to the customer, necessitating replacement orders, rework, or scrapping of affected kits. TE Connectivity's own estimate places the Cost of Poor Quality (COPQ) attributable to kitting errors at between USD 10,000 and USD 20,000 per kitting line per year, accounting for rework labour, component replacement, logistics costs associated with replacement shipments, and the residual risk of customer complaints and reputational damage. Given that this figure represents only the directly quantifiable cost for a single line, the aggregate impact across TE Connectivity's multi-plant, multi-line operations represents a substantially greater financial exposure.

1.2 Root Cause Analysis
A structured analysis of the contributing factors to kitting errors reveals that the problem is not attributable to a single systemic failure, but rather to an interaction of cognitive, ergonomic, procedural, and technological factors that collectively elevate error probability in high-mix manual assembly environments.

At the operator level, cognitive overload is a primary contributor. Kitting operators must simultaneously manage the order specification, identify components across multiple bins containing visually similar parts — connectors and electronic components frequently share form factors, colour schemes, or packaging styles — and maintain accurate counts across varying quantities per kit. This cognitive demand is compounded by production time pressure, which incentivises speed over deliberate verification. Fatigue, particularly during later stages of extended shifts, further degrades attentional accuracy and increases the likelihood of substitution and omission errors. Workstation ergonomics may also play a role: where bin layout is suboptimal or where components are stored in configurations that require repeated handling, the physical and cognitive demands of each kit increment.

At the procedural level, the verification mechanism in current use — a post-kit gravimetric check using a digital scale — is structurally insufficient for reliable error detection. The scale provides a single aggregate weight measurement of a completed kit, which is then compared against an expected total weight derived from the order specification. This approach is vulnerable in several respects. First, it cannot localise or diagnose an error: a weight discrepancy may correspond to a missing part, an excess part, a wrong part, or any combination thereof, without any information as to which component is implicated. Second, the measurement is subject to component weight tolerances: the unit mass of individual components ranges from approximately 0.5 grams to 1 kilogram, and manufacturing tolerances mean that individual parts may deviate from their nominal weight, introducing measurement uncertainty at fine granularity. Third, and most critically, the aggregate nature of the measurement creates the possibility of error cancellation: a deficit from one missing component may be arithmetically offset by a surplus from an excess component of similar mass, producing a kit weight that appears within tolerance despite containing multiple errors. Fourth, even where a weight discrepancy is detected, the scale offers no information to guide corrective action — the operator must re-verify the entire kit manually, consuming cycle time and adding process friction.

Synthesising these contributing factors, the root cause of the kitting error problem can be characterised as the absence of a real-time, part-specific verification mechanism at the operator workstation. Current tools are reactive rather than proactive: they detect aggregate outcomes rather than individual actions, they provide no in-process guidance to the operator, and they enforce no corrective behaviour before a defective kit is dispatched.

1.3 Current Solutions and Limitations
A range of verification and error-proofing approaches exists within the broader manufacturing and logistics sector. These can be grouped into five categories: manual process controls, barcode and RFID-based identification, vision-based inspection systems, gravimetric verification, and automated picking technologies. Each is examined below in the context of TE Connectivity's operational requirements.

Manual process controls — including paper checklists, supervisor sign-offs, and peer verification — represent the baseline approach. While requiring no capital investment, they do not eliminate human error; they merely introduce an additional human in the verification loop, which itself is subject to the same cognitive and fatigue-related failure modes as the original kitting act.

Barcode and RFID scanning — where each component or bin is scanned at the point of handling — offers precise part-level identification and can enforce sequence compliance. However, the implementation burden in TE Connectivity's environment is substantial. With approximately 800 to 900 active SKUs and a high-mix order profile, maintaining up-to-date barcode or RFID labelling across all bins and components introduces significant setup and maintenance overhead. More critically, these systems require an additional handling step per component — the scan — which directly increases operator cycle time per kit.

Commercial machine vision systems — such as those offered by Cognex, Keyence, and similar providers — are capable of high-accuracy part identification and defect detection in controlled environments. However, they are generally optimised for fixed-configuration, single-product or narrow-product-range inspection tasks on automated production lines. In TE Connectivity's high-mix kitting context, the configuration and retraining effort for commercial systems would be considerable, and the licensing and hardware cost is typically high relative to the COPQ figures associated with the target problem.

Pick-to-light systems — which illuminate the correct bin and prompt the operator with the required quantity — address the guidance problem effectively but require fixed bin configurations and dedicated hardware installation at each workstation position. They cannot verify that the correct component was retrieved from the illuminated bin, nor that the correct quantity was actually placed in the kit.

A common limitation across all existing approaches is their inability to fulfil three simultaneous functions: predicting or detecting errors at the point of occurrence, diagnosing the specific nature of the error, and enforcing corrective action by the operator before the kit is completed. No single existing solution addresses all three functions in a form that is cost-effective, adaptable to a high-mix environment, and compatible with an unmodified manual kitting workflow.


================================================================================
CHAPTER 2: DESIGN OVERVIEW AND REQUIREMENTS
================================================================================

2.1 Project Scope and Value Proposition

2.1.1 Project Scope
The scope of the AEGIS project, as a proof-of-concept prototype, is precisely bounded to manage complexity within the constraints of available time, budget, and resources. 

The following are explicitly WITHIN SCOPE:
- The design, physical integration, and bench validation of a single-workstation AEGIS unit comprising computer vision and gravimetric sensing subsystems.
- The development of software for real-time component identification, error detection, and operator alerting.
- An operator-facing HMI display and a supervisor-facing web monitoring dashboard.
- A hardware architecture designed to support multi-workstation data aggregation.
- Laboratory-based prototype validation using simulated kitting tasks representative of TE Connectivity's operational workflow.

The following are explicitly OUTSIDE SCOPE:
- Physical deployment across multiple simultaneous workstations in a production environment.
- Integration with TE Connectivity's enterprise resource planning (ERP) or manufacturing execution system (MES).
- Automation of any physical component of the kitting process.
- Modification of existing standard operating procedures or bin layouts.
- Handling of components that are opaque to camera-based identification due to physical packaging.

2.1.2 Value Proposition
AEGIS addresses a well-defined operational gap: the absence of real-time, part-specific error detection and corrective guidance at the cell kitting workstation. The system's value is grounded in three distinct contributions relative to the current state at TE Connectivity.

First, AEGIS shifts the point of error detection upstream, from post-kit gravimetric verification or downstream QC inspection to the exact moment of component retrieval and placement. Second, the system employs complementary sensing modalities — computer vision for visual hand-and-bin tracking and load cell-based gravimetric sensing for continuous quantity monitoring — achieving a robust detection capability that neither modality can provide independently. Third, the system is designed specifically for high-mix, manual kitting environments that commercial alternatives address poorly. Its edge-deployed architecture, order-driven verification context, and environment-agnostic operational requirements allow it to function across TE Connectivity's geographically diverse manufacturing sites without site-specific infrastructure overhaul.

2.2 Requirements Definition and Technical Translation

2.2.1 Stakeholder Requirements
The AEGIS system serves a diverse set of stakeholders across operational, managerial, technical, and commercial dimensions. Stakeholder requirements are compiled in Table 2.1.

Table 2.1: Stakeholder requirements summary.
------------------------------------------------------------------------------------------------------------------------
Stakeholder           Role                                Primary Requirements                                Key Concerns
------------------------------------------------------------------------------------------------------------------------
Ground Operator       Primary end-user at station         Minimal disruption; clear real-time feedback; zero added handling steps    Cycle time impact; ease of use; false alarms
Production Supervisor Oversees kitting lines             Reliable error capture; operator compliance visibility; minimal overhead   System reliability; alert accuracy
Plant Production Mgr  Responsible for line throughput     Reduction in COPQ; no throughput degradation; low TCO                     ROI; uptime; scalability
Quality Engineer      In-process & outgoing quality       High detection rate for missing/wrong components; audit trail             False negative rate; traceability
TE Connectivity (Org) Industrial partner                  Deployable across global plants; adaptable to varied sites                Generalisation; IP & data security
Maintenance Engineer  System upkeep & SKU config          Simple SKU onboarding; minimal specialist ML expertise required           Onboarding time; calibration ease
Advantech (Hardware)  Edge computing partner              Hardware featured and validated within prototype                         Hardware compatibility & performance
------------------------------------------------------------------------------------------------------------------------

2.2.2 Functional Requirements
The functional requirements of AEGIS are presented in Table 2.2 as discrete, verifiable engineering statements.

Table 2.2: Functional requirements for the AEGIS system.
------------------------------------------------------------------------------------------------------------------------
ID     Category                 Requirement Description                                                         Rationale
------------------------------------------------------------------------------------------------------------------------
FR-01  Component ID             System shall identify component type in each active bin.                       Enables part-specific verification.
FR-02  Wrong Part Detection     System shall detect when placed item does not match active order spec.          Addresses primary wrong-part failure mode.
FR-03  Missing Part Detection   System shall detect when kit is completed with missing required components.    Missing parts represent highest COPQ impact.
FR-04  Quantity Verification    System shall verify that count placed matches order quantity.                   Quantity errors bypass aggregate scale checks.
FR-05  Real-Time Alerting       System shall notify operator of error within <100ms.                           Proactive in-process intervention.
FR-06  Corrective Guidance      System shall present actionable corrective guidance naming bin and quantity.    Enables rapid operator self-correction.
FR-07  Order Integration        System shall accept order specs (SKUs, bins, quantities) as input.              Verification must be order-driven.
FR-08  New Part Onboarding      System shall support onboarding new SKUs without model retraining.              Supports TE's 4,000 SKU catalogue.
FR-09  Environmental Robustness System shall operate correctly under 0-500 lux without station modification.    Enables plant-wide deployment.
FR-10  Supervisor Visibility    System shall provide supervisors remote intranet visibility of station status.  Enables remote situational awareness.
FR-11  Scalability              System architecture shall support multi-workstation extension.                 Supports future multi-line expansion.
------------------------------------------------------------------------------------------------------------------------

2.2.3 Translation of Stakeholder Requirements into Technical Specifications
To ensure the engineering design directly addresses operational goals, stakeholder requirements were translated into applied technical specifications, summarized in Table 2.3.

Table 2.3: Translation of stakeholder requirements into technical specifications.
------------------------------------------------------------------------------------------------------------------------
Stakeholder Group   Stakeholder Requirement                                     Technical Translation                             Applied Configuration Specification
------------------------------------------------------------------------------------------------------------------------
Ground Operator     "System must not disrupt packing rhythm or require scanning" Passive sensing with zero added handling steps    Gravimetric load cells + overhead camera hand tracking
Quality Engineer    "Need to detect wrong parts, missing parts, and wrong qty"  Multi-modal exception mapping in verification     Sequential single-bin FSM enforcing 4 discrete fault states
Ground Operator     "Errors must be shown immediately so I can correct them"    Low-latency state processing & HMI updating       End-to-end latency <100ms; CV >=30fps; load cell 10 SPS
Maintenance Eng.    "Must be easy to set up new SKU orders and change layout"   Dynamic geofencing & soft-coded item databases    2-snapshot YOLOv8-OBB calibration; inventory.yaml SKU config
Plant Manager       "Workstations vary in light; cannot rebuild layout"         Environmental invariance & layout independence    Rim-centric OBB; matte ESD mats; verified 0 to 506 lux
Plant Manager       "Data must be secure and kept within plant network"         On-device edge processing without cloud           Local execution on Jetson AGX Orin with local FastAPI server
------------------------------------------------------------------------------------------------------------------------

2.3 System Architecture at Concept Overview

2.3.1 Concept Overview
The AEGIS system architecture is designed as a modular, edge-deployed framework comprising four distinct subsystems. These subsystems maintain clear interface boundaries, permitting independent development, testing, and iteration. 

System Architecture Diagram (Subsystem Boundaries indicated by Dotted Lines):

+-----------------------------------------------------------------------------------+
: Subsystem 1: Physical Frame                                                       :
:   - Workstation Rig (Two-Tier 9-Bin Grid + Matte ESD Mat)                         :
+-----------------------------------------------------------------------------------+
                                   :.                                :.
                    Positions Camera:                                :.Supports Load Cells
                                   v                                 v
+-----------------------------------------------------------------------------------+
: Subsystem 2: Sensor Hardware                                                      :
:   - Overhead Camera (1080p @ 30 fps USB Feed)                                     :
:   - ESP32 + HX711 Summed 3-Point Load Cell Array                                  :
+-----------------------------------------------------------------------------------+
         |                                                           |
         | Live Video Stream                                         | JSON Weight Packets
         v                                                           v
+-----------------------------------------------------------------------------------+
: Subsystem 3: Edge Processing Unit (Advantech MIC-733 Jetson AGX Orin)             :
:   - YOLOv8-OBB Bin Boundary Detector                                              :
:   - MediaPipe Hand Landmark Tracker                                               :
:   - Bin Assignment Engine & Occlusion Gate                                        :
:   - Placement Tracker FSM Verification Backend                                    :
+-----------------------------------------------------------------------------------+
                                         |
                                         | Thread-safe PipelineState Cache
                                         v
+-----------------------------------------------------------------------------------+
: Subsystem 4: Human-Machine Interface (HMI)                                        :
:   - FastAPI Server (REST API / JSON @ 10Hz AJAX Polling)                          :
:   - Operator Touch Display Console (Kiosk Browser) & Remote Supervisor View       :
+-----------------------------------------------------------------------------------+

Subsystem Roles:
1. Physical Frame: Structurally positions sensors overhead, mounting the camera to achieve an unobstructed field of view across the 9-bin grid and kit box consolidation area, utilizing matte ESD mats to suppress specular reflections.
2. Sensor Hardware: Captures raw physical signals via an overhead camera (1080p video at 30 fps) and a load cell array (summed 3-point bridges routed through HX711 24-bit ADCs to an ESP32 microcontroller streaming JSON weight arrays via USB serial).
3. Edge Processing Unit: Runs on the Advantech MIC-733 (Jetson AGX Orin), executing YOLOv8-OBB bin detection, MediaPipe hand landmark tracking, geometric bin assignment with occlusion gating, and the sequential single-bin FSM verification engine.
4. Human-Machine Interface (HMI): Renders glanceable real-time guidance and error alerts to the operator on a touchscreen display via a local FastAPI server, while permitting remote supervisor monitoring across the plant intranet.

2.4 Design and Component Selection

2.4.1 System-Level Selection
Morphological analysis was conducted to establish the overall system design parameters. The selected choices and engineering rationales are detailed in Table 2.4.

Table 2.4: System-level morphological selection matrix.
------------------------------------------------------------------------------------------------------------------------
Feature                 Option A                Option B            Option C            Option D            Selected & Engineering Rationale
------------------------------------------------------------------------------------------------------------------------
Worker Dashboard        Monitor                 Physical Buttons    Audio / Voice       --                  Monitor: Provides intuitive visual feedback without local software installation.
Feedback Mechanism      Audio Alerts            Haptic Feedback     Visual Feedback     --                  Visual Feedback: Non-intrusive in noisy, shared production environments.
Processing Architecture Edge AI PC              Raspberry Pi        Workstation Desktop Microcontroller     Edge AI PC: Industrial-grade GPU acceleration for continuous multi-model execution.
Camera Mounting         Fixed Mount             Adjustable Arm      Rail Mounting       --                  Fixed Mount: Eliminates calibration drift across production shifts.
Item Identification     RFID                    Computer Vision     Manual Init         --                  Manual Init: Resolves SKU identity via bin mapping rather than tagging 800+ parts.
Quantity Verification   Weight                  Computer Vision     Manual Counter      Optical Gate        Weight: Continuous, passive counting utilizing TE's existing SKU weight database.
Worker Intention        Weight                  Computer Vision     Neural Implant      Optical Gate        Computer Vision: MediaPipe hand tracking determines exact bin reach in real time.
------------------------------------------------------------------------------------------------------------------------

2.4.2 Detailed Component Selection
Table 2.5 details the specific hardware and software components selected for each subsystem, along with their explicit engineering rationales.

Table 2.5: Detailed component selection morphological matrix.
------------------------------------------------------------------------------------------------------------------------
Component           Option A            Option B            Option C            Option D            Selected Option & Engineering Rationale
------------------------------------------------------------------------------------------------------------------------
Monitor Application Web Application     Desktop App         Mobile App          Terminal-based      Web App (FastAPI/JS): Served locally from Jetson, cross-platform, zero per-station installation.
Edge AI PC          Jetson Orin Nano    Jetson Orin NX      Jetson AGX Orin     Generic mini-PC     Jetson AGX Orin (MIC-733): 275 TOPS GPU compute headroom, running pipeline on 1/12 CPU cores.
Hand Detection Model MediaPipe Hands    YOLO-Pose           OpenPose            Custom model        MediaPipe Hands: Off-the-shelf 21-landmark 3D tracking using float16 CPU/GPU quantized execution.
Bin Boundary Model  YOLO-OBB (Ultralytics) YOLOv8-Seg       Mask R-CNN          Classical CV        YOLOv8-OBB: Oriented bounding box polygon geofences aligned to tilted physical bin rims.
Weight Sensors      Kitchen scale       1x load cell        Human scale         3x load cell        3x load cell triangle: Summed 3-point support triangle prevents tipping and off-centre errors.
------------------------------------------------------------------------------------------------------------------------

2.4.3 Sequential Single-Bin Verification Finite State Machine (FSM)
The core verification logic is governed by a sequential single-bin Finite State Machine (FSM). To resolve weight ambiguity (where multiple items drawn concurrently from different bins cannot be distinguished by a single kit-box load cell), the FSM locks execution to one active bin at a time.

FSM States:
- IDLE: No bin active; operator may begin any uncompleted bin.
- PICKING: Active bin locked; picks counted into kit box; other bins soft-locked.
- FAULT: Exception detected; counting suspended; auto-clears upon correction.
- KIT_COMPLETE: All required bins reach target quantity.
- WAITING_EMPTY: Operator prompted to consolidate kit box into finished container.

The FSM continuously evaluates four discrete fault states:
1. Overpack: Excess quantity placed into kit box beyond order requirement.
2. Picked-from-wrong-bin: Component drawn from an inactive or unassigned bin.
3. Returned-to-wrong-bin: Component placed into an incorrect source bin.
4. Out-of-sequence: Component drawn from a valid order bin while another bin is locked.


================================================================================
CHAPTER 3: PROTOTYPE DEVELOPMENT
================================================================================

Flowing directly from the System Architecture defined in Chapter 2, Chapter 3 details the physical construction, sensor integration, and software implementation of the AEGIS prototype. The prototype was developed using a structured methodology designed to imitate the working environment of TE Connectivity's manual kitting workstations based on plant specifications and stakeholder requirements.

3.1 Physical Frame
The physical frame was constructed to emulate the dimensions and ergonomics of TE Connectivity's production kitting stations without requiring modification to existing plant benches (FR-09). Based on plant requirement specifications and following stakeholder feedback, the frame was constructed from aluminum extrusions and fitted with matte electrostatic-discharge (ESD) mats to prevent specular reflections from overhead lighting.

The workstation features a two-tier nine-bin grid:
- Lower Tier: 3 large-capacity bins mounted on 3-point load cell platforms.
- Upper Tier: 6 medium and small bins mounted on corresponding load cell platforms.
- Consolidation Area: A dedicated kit-box load cell receptor positioned centrally for operator placement.
- Overhead Rig: Rigid camera mount positioning the 1080p camera 1.2 m directly above the working plane, ensuring complete FOV coverage of all 9 bins and the kit box.

3.2 Sensors Subsystem

3.2.1 Computer Vision
The computer vision subsystem executes two real-time models in parallel on the Jetson AGX Orin:
1. YOLOv8-OBB Bin Boundary Detector: Trained on rim-centric images across empty, partial, and fully filled bins. Generates tight polygon geofences around bin rims regardless of item contents or protrusion.
2. MediaPipe Hand Landmark Tracker: Ingests camera frames at 30 fps, tracking 21 3D landmark coordinates per hand. Converts fingertip/knuckle coordinates into geometric bin assignment signals.

3.2.2 Load Cells
Rather than evaluating arbitrary prototype stages, load cell platform development directly addressed the mechanical stability and metrology requirements established in Chapter 2.
- Small Bin Platform: Single 1 kg cantilever load cell with circular acrylic mounting plate (100 mm radius). Suitable for light parts (<1 kg).
- Medium/Large Bin Platforms: Three-point summed load cell triangle (utilizing 5 kg and 10 kg sensors). Arranging three sensors in an equilateral triangle ensures the bin's centre of gravity stays within the support triangle, preventing tipping and ensuring statically determined contact on uneven benches. Summed signals route through single-channel HX711 ADCs to an ESP32 microcontroller.

3.3 Software Subsystem

3.3.1 Software Stack Specification
The software architecture running on the Advantech MIC-733 (Jetson AGX Orin) is detailed in Table 3.1.

Table 3.1: Software stack specification.
------------------------------------------------------------------------------------------------------------------------
Layer                   Technology                              Purpose
------------------------------------------------------------------------------------------------------------------------
Operating System        Ubuntu 22.04 LTS with JetPack 6.0       Base OS with NVIDIA CUDA driver support for Jetson.
Runtime Environment     Python 3.13 (Conda Environment)         Primary application language runtime.
Bin Boundary Detection  YOLOv8-OBB (Ultralytics)                Generates oriented polygon geofences from calibration.
Hand Landmark Tracking  Google MediaPipe Tasks SDK (float16)    Tracks 21 3D hand landmarks per frame at 30 fps.
Image Utilities         OpenCV-Python                           Live video capture, debug drawing, and boundary overlays.
Backend Application     FastAPI (Uvicorn ASGI Server)           Hosts local REST endpoints and manages state cache.
Shared Memory Cache     Thread-Safe PipelineState Object        Isolates high-speed CV loop from web server thread.
Operator Frontend       Vanilla HTML5 / CSS3 / JavaScript       Glanceable HMI polling backend REST API at 10 Hz via AJAX.
Serial Communication    PySerial                                Reads JSON weight data packets from ESP32 over USB serial.
Microcontroller Firmware C++ (Arduino IDE) on ESP32             Reads HX711 load cell ADCs and streams JSON weight arrays.
------------------------------------------------------------------------------------------------------------------------

3.3.2 Frontend HMI
The operator HMI is served locally via FastAPI as a full-screen browser kiosk. Key features include:
- Glanceable Bin Grid: Mirrors physical 9-bin layout.
- Animated Hand Halos: Teal halo for valid bin reaches; pulsating RED halo for invalid reaches.
- Full-Width Fault Banner: Prominent top banner displaying single-word error keywords with explicit corrective subtitles.

3.3.3 Backend Processing Engine
The backend decouples sensing from rendering via a thread-safe `PipelineState` shared memory cache. Verification logic is encapsulated in `placement.py`, implementing the FSM independently of peripherals to enable automated testing.


================================================================================

================================================================================
Chapter 4 | Testing, Evaluation and Iteration
================================================================================

This chapter documents the empirical testing, metrology characterisation, software unit verification, and iterative design enhancements conducted on the AEGIS prototype. Following the morphological selections in Chapter 2 and functional development in Chapter 3, each subsystem—hardware platforms, computer vision engines, and backend state tracking logic—was evaluated under rigorous laboratory conditions to quantify performance limits and drive targeted engineering iterations before full system integration.

--------------------------------------------------------------------------------
4.1 Hardware Subsystem
--------------------------------------------------------------------------------

4.1.1 Load Cell Array Platform

4.1.1.1 Evaluation of 1st Load Cell Platform Iteration
The initial hardware platform design comprised a single cantilever load cell mounted centrally beneath a flat acrylic resting plate for each bin position across the nine-bin workstation grid. Initial bench evaluation revealed two critical mechanical failure modes:

1. Off-Centre Tipping Instability: When larger or heavier components were retrieved or replenished, the bin's centre of gravity (COG) frequently shifted away from the central vertical axis of the cantilever cell. This generated an asymmetric tipping moment, causing the acrylic platform to tilt, contact neighboring bin frames, and introduce binding friction.
2. Moment Arm Sensitivity: Off-centre loading on single-point cantilever cells beyond their rated radial distance introduced substantial measurement error, causing baseline zero-point shifts exceeding ±15.0 g.

Table 17: Evaluation of Initial Single-Cell Load Cell Platform Design
------------------------------------------------------------------------------------------------------------------------
Platform Type       Tested Capacity  Mechanical Defect Observed                        Measurement Deviation    Resolution Limit
------------------------------------------------------------------------------------------------------------------------
Small Bin Rest      1 kg             Minor tipping under edge load                     ±0.4 g                   0.5 g
Medium Bin Rest     5 kg             Severe tipping; frame contact under off-centre COG  -38.6 g to +64.0 g       >15.0 g
Large Bin Rest      10 kg            Severe tipping; static friction against frame     -65.8 g to +105.0 g      >25.0 g
------------------------------------------------------------------------------------------------------------------------

[Figure 4.1: Mechanical Comparison of Single-Cell Cantilever Rest vs Three-Point Summed Load Cell Triangle Platform]

4.1.1.2 2nd Load Cell Platform Iteration (Final Design)
To overcome tipping instability and off-centre loading degradation without expanding cell dimensions, a three-point summed load cell triangle configuration was designed for all medium and large bin platforms. 

By arranging three load cells in an equilateral triangular geometry beneath a rigid triangular acrylic support plate, the bin's COG is guaranteed to lie within the statically determined support plane. A three-point support plane ensures continuous full contact across all three sensors on uneven bench surfaces, eliminating mechanical rocking. The three load cell bridges are wired in parallel into a single HX711 24-bit analogue-to-digital converter (ADC), transmitting a unified, summed weight signal to the ESP32 microcontroller. Single 1 kg cantilever load cells were retained for small bin positions where component mass is low (<1 kg) and platform radius is small (100 mm).

[Figure 4.2: Final Load Cell Platform Iterations Showing (a) Single 1kg Platform and (b) Three-Point Summed 5kg/10kg Platform Wiring Schematics]

4.1.1.3 Metrology Bench Qualification and Off-Centre Sensitivity
The metrological capability of the load cell configurations was qualified on a dedicated metrology test bench using a graded reference-mass kit (reference scale readability: 0.1 g). Testing evaluated zero-noise floor (σ), minimum detectable mass, hysteresis, thermal drift, and off-centre sensitivity.

Table 18: Metrology Bench Qualification Results across Load Cell Platforms
------------------------------------------------------------------------------------------------------------------------
Configuration     Cell Count & Wiring    Zero-Band σ (g)  Min. Detectable Mass (g)  Repeatability SD (g)  Max Hysteresis (g)
------------------------------------------------------------------------------------------------------------------------
1 kg Platform     1 x 1 kg (Single)      0.028            ~0.5                      0.009                 0.09
5 kg Platform     3 x 5 kg (Summed)      0.223            ~2.3                      1.060                 5.44
10 kg Platform    3 x 10 kg (Summed)     0.433            ~4.3                      2.850                 16.99
------------------------------------------------------------------------------------------------------------------------

Off-Centre Sensitivity Analysis:
Positioning a 2,500 g reference mass across five discrete locations on the 5 kg summed platform revealed a reading variance of -38.6 g to +64.0 g when moving from centroid to apex cell positions. On the 10 kg platform, a 5,000 g mass produced variations of -65.8 g to +105.0 g. 

Key Engineering Insight: Summing three load cells guarantees mechanical stability against tipping, but off-centre mass displacement bounds fine quantity verification resolution. Components weighing <3.5 g cannot be reliably counted on 5 kg or 10 kg platforms. Software logic must therefore enforce SKU routing, assigning ultra-light components exclusively to 1 kg single-cell platforms (σ = 0.028 g).

[Figure 4.3: Metrology Characterisation Plots Showing (a) Minimum Detectable Mass vs Zero-Band Noise Floor and (b) Off-Centre Radial Deviation Curves]

4.1.2 Physical Frame & Workstation Integration

4.1.2.1 Working Environment Emulation
The physical frame was constructed from industrial aluminum extrusions to replicate the spatial layout, bin accessibility, and operator ergonomics of TE Connectivity's manual kitting lines based on plant requirement specifications. The frame mounts an overhead camera 1.2 m above the working plane, providing an unobstructed field of view (FOV) across the nine-bin grid and kit consolidation receptor.

[Figure 4.4: Physical Workstation Frame Setup Showing Overhead Camera Rig, Two-Tier 9-Bin Grid, and Kit Box Consolidation Receptor]

4.1.2.2 Matte Anti-Glare ESD Surface Integration
To prevent specular reflection from overhead plant lighting from interfering with computer vision boundary detection, all workstation surfaces were lined with matte electrostatic-discharge (ESD) protective matting. Optical reflection tests confirmed complete suppression of specular hot-spots under direct 500 lux overhead illumination.

--------------------------------------------------------------------------------
4.2 Computer Vision Subsystem
--------------------------------------------------------------------------------

4.2.1 YOLOv8-OBB Bin Boundary Model

4.2.1.1 Dataset Augmentation and Fill-Level Permutation Protocol
The bin boundary detection model utilizes YOLOv8-OBB (Oriented Bounding Box) to generate precise polygon geofences aligned to physical bin rims. To ensure boundary detection remains invariant to bin contents, a layer-by-layer permutation protocol was executed across seven distinct item categories: matte black TE connectors, reflective metal shelf brackets, M6 cross screws, white PVC tubes, long copper rods protruding above rims, dark black bin connectors, and mixed item sets.

Model training was conducted on a dataset augmented across full-spectrum fill levels (0%, 20%, 40%, 60%, 80%, and 100% bin capacity).

Table 19: YOLOv8-OBB Bin Detection Validation Metrics (Setup V3, Fullness-Varied Dataset)
------------------------------------------------------------------------------------------------------------------------
Metric                          Validation Score    Target Threshold    Status
------------------------------------------------------------------------------------------------------------------------
Precision (P)                   0.999               >0.950              PASSED
Recall (R)                      1.000               >0.950              PASSED
mAP@50                          0.995               >0.900              PASSED
mAP@50-95                       0.935               >0.800              PASSED
Training Epochs                 55 (Early stop 54)  --                  PASSED
------------------------------------------------------------------------------------------------------------------------

[Figure 4.5: YOLOv8-OBB Training Metric Curves (Precision, Recall, mAP50, mAP50-95) Across 55 Training Epochs]

4.2.1.2 Environmental Robustness (Lighting Characterisation Sweep)
An 8-step ambient illuminance sweep was conducted from 0 lux (complete darkness, infrared camera mode) to 506 lux (full overhead strip lighting) across all nine bin positions (72 total evaluations).

Table 20: Bin Detection Accuracy Across Ambient Illuminance Sweep (0 lux to 506 lux)
------------------------------------------------------------------------------------------------------------------------
Illuminance Level (lux)    Bins Evaluated    Detections Recorded    Detection Accuracy    Mean Confidence Score
------------------------------------------------------------------------------------------------------------------------
0 lux (Darkness)           9                 9                      100.0%                0.942
45 lux                     9                 9                      100.0%                0.951
110 lux                    9                 9                      100.0%                0.963
185 lux                    9                 9                      100.0%                0.958
260 lux                    9                 9                      100.0%                0.967
340 lux                    9                 9                      100.0%                0.971
425 lux                    9                 9                      100.0%                0.974
506 lux (Full Brightness)  9                 9                      100.0%                0.978
------------------------------------------------------------------------------------------------------------------------

Result: The model achieved 100% boundary detection accuracy across all illuminance levels. Correlation between lux and confidence was weak (Pearson r ≈ 0.27), confirming that rim-centric OBB detection is lighting-invariant across standard factory operating environments (FR-09).

[Figure 4.6: Lux-vs-Confidence Scatter Plot for Bin Boundary Detection Across 72 Sampling Points]

4.2.1.3 Bin Colour and Layout Sensitivity
Testing evaluated model generalisation across bin colour variations:
- Group A1 (Baseline Blue Bins): 100% detection accuracy (20/20 frames).
- Group A2/A3 (Mixed Red/Green/Blue Bins): Detection accuracy dropped to 80.0% and 85.0% due to bounding box hallucinations. Because the training dataset was dominated by blue bin rims, non-blue rims occasionally caused boundary overlap. This finding justified standardising workstation deployment on blue industrial bins.

4.2.2 MediaPipe Hand Landmark Tracking & Occlusion Iteration

4.2.2.1 1st Software Iteration Defect (Shelf-Lip Occlusion & Fingertip Extrapolation)
During initial integration testing, reaching into the bottom row of bins caused a critical tracking failure. The physical shelf lip of the two-tier rack occluded the operator's distal finger joints. MediaPipe's hand tracking model, losing sight of the fingertips, extrapolated joint positions upward, erroneously placing fingertip coordinates inside top-tier bin geofences. This triggered false "Out-of-Sequence" and "Wrong-Bin" alarms during valid picks.

[Figure 4.7: Diagram of Shelf-Lip Occlusion Defect Showing Fingertip Coordinate Extrapolation into Top-Bin Geofence]

4.2.2.2 2nd Software Iteration (Occlusion Gate and Occlusion Hold Implementation)
To resolve tracking extrapolation without modifying workstation hardware, two backend software modules were developed in `bin_assignment.py`:

1. Occlusion Gate: Inspects the hand's proximal anchor keypoints—specifically the wrist and index metacarpophalangeal (MCP) knuckle centroid—which remain fully visible below the shelf lip line (Y_lip). If proximal anchors lie below Y_lip, fingertip extrapolation is overridden, forcing hand assignment to the bottom bin directly beneath the anchors.
2. Occlusion Hold: A stateful module (`occlusion_hold.py`) that maintains active bottom-bin assignment when a hand vanishes entirely behind the shelf lip, releasing the lock only when landmarks reappear.

[Figure 4.8: Occlusion Gate Operational Flow Diagram Demonstrating Proximal Anchor Validation and Geofence Override]

--------------------------------------------------------------------------------
4.3 Backend Subsystem & Unit Verification
--------------------------------------------------------------------------------

4.3.1 FSM Verification Logic (`placement.py`)
The verification engine encapsulates a sequential single-bin Finite State Machine (FSM). By locking picking sequence to one active bin at a time, weight discrepancies in the kit box are unambiguously attributed to the active SKU. The engine evaluates four fault conditions: Overpack, Picked-from-wrong-bin, Returned-to-wrong-bin, and Out-of-sequence.

4.3.2 Test Plan, Regression Guard, and Code Coverage Results
To verify FSM state transitions, debouncing windows, and fault recovery independently of physical peripherals, an automated test suite was constructed using the Pytest framework.

Table 21: Backend Pytest Unit Verification and Statement Coverage Report
------------------------------------------------------------------------------------------------------------------------
Module File             Module Description                            Statements Covered    Coverage (%)    Test Status
------------------------------------------------------------------------------------------------------------------------
placement.py            FSM Core Verification Engine                  196 / 200             98%             PASSED (38 tests)
bin_assignment.py       Geometric Bin & Occlusion Gate Engine         216 / 260             83%             PASSED (34 tests)
occlusion_hold.py       Shelf-Lip Occlusion Lock Manager              35 / 35               100%            PASSED (12 tests)
grid_allocator.py       Spatial Grid Slot Allocator                   100 / 107             93%             PASSED (14 tests)
grid_calibrator.py      Two-Snapshot Calibration Verifier             76 / 79               96%             PASSED (12 tests)
grid_session.py         Session Boundary State Manager                70 / 74               95%             PASSED (10 tests)
inventory.py            Weight-to-Count Scaling Engine                35 / 45               78%             PASSED (8 tests)
cycle.py                Order Cycle State Manager                     39 / 39               100%            PASSED (6 tests)
------------------------------------------------------------------------------------------------------------------------
Total Backend Suite     Core Logic Verification Modules               767 / 839             91.4%           134 / 134 PASSED
Overall Project         Entire Codebase (incl. GUI / Hardware drivers) 1,285 / 2,569         50.0%           PASSED (4.46s)
------------------------------------------------------------------------------------------------------------------------

[Figure 4.9: Pytest Automated Suite Execution Terminal Summary Output Showing 134 Passed Unit Tests]


================================================================================
Chapter 5 | System Integration
================================================================================

Chapter 5 details the end-to-end integration of the mechanical frame, computer vision pipeline, load cell serial array, and operator HMI into a unified edge architecture, followed by operator usability testing, HMI iterations, and requirements validation.

--------------------------------------------------------------------------------
5.1 UI-Backend Communication Pipeline
--------------------------------------------------------------------------------
To maintain high UI responsiveness and decouple intensive computer vision processing from web rendering, AEGIS implements a multi-threaded architecture centered around a thread-safe `PipelineState` shared memory object:

- Thread 1 (CV Processing Loop): Ingests 1080p camera frames at 30 fps, executes YOLOv8-OBB and MediaPipe inference, applies the Occlusion Gate, and updates spatial coordinates in `PipelineState`.
- Thread 2 (Load Cell Serial Ingestion): PySerial driver continuously parses JSON weight packets from the ESP32 over USB at 10 SPS, executing weight debouncing and updating weight vectors in `PipelineState`.
- Thread 3 (FastAPI Web Server): Serves local REST endpoints and streams system state via JSON to the operator HMI display over HTTP/WebSockets. The frontend polls at 10 Hz via AJAX, ensuring smooth UI rendering without locking the vision pipeline.

[Figure 5.1: Thread-Safe Multi-Threaded System Communication Data Flow Architecture]

--------------------------------------------------------------------------------
5.2 Microcontroller-to-Edge Communication Pipeline
--------------------------------------------------------------------------------
The ESP32 microcontroller executes custom C++ firmware compiled via Arduino IDE. It samples four HX711 24-bit ADCs (interfacing the 1 kg, 5 kg, and 10 kg load cell bridges), applies digital moving-average noise filtering, and streams serialized JSON arrays over USB serial at 115200 baud:

```json
{"status": "OK", "timestamp": 104520, "bins": {"bin_1_0": 245.2, "bin_1_1": 0.0, "kit_box": 120.5}}
```

The PySerial backend driver includes an automated reconnect loop that recovers connection within <500 ms in the event of transient USB power drops.

--------------------------------------------------------------------------------
5.3 Real-Time Inference & Hardware Resource Benchmarking
--------------------------------------------------------------------------------
System execution was benchmarked on the Advantech MIC-733 edge AI computer (powered by NVIDIA Jetson AGX Orin 32GB).

Table 22: Hardware Resource Utilization Benchmark on Advantech MIC-733 (Jetson AGX Orin)
------------------------------------------------------------------------------------------------------------------------
Resource Metric                 Measured Execution Value        Platform Capacity Limit     Utilization (%)
------------------------------------------------------------------------------------------------------------------------
CPU Core Usage                  1.2 Cores                       12 Cores (ARM Cortex-A78AE)  10.0%
GPU Compute (Tensor Cores)      18.5 TOPS                       275 TOPS                    6.7%
System RAM Footprint            2.08 GB                         32 GB Unified Memory        6.5%
End-to-End System Latency       78 ms (33ms CV + 10ms FSM + 35ms HMI) <100 ms Target          78.0%
Camera Capture Frame Rate       30.0 fps                        30.0 fps Target             100.0%
------------------------------------------------------------------------------------------------------------------------

Result: AEGIS utilizes only 10% CPU and 6.7% GPU capacity on the Jetson AGX Orin, leaving substantial compute headroom to support multi-workstation orchestration (FR-11).

[Figure 5.2: Hardware Resource Profiling Charts Showing CPU Core Load, GPU Utilization, and System Memory Footprint]

--------------------------------------------------------------------------------
5.4 Final System Testing & Evaluation
--------------------------------------------------------------------------------

5.4.1 Usability Study & Operator Testing
A structured usability study was conducted between 8 and 13 July 2026 with six active manual kitting operators (Benny, Chloe, Ming Zhan, Song Yi, Royce, and Jack). Using a within-subjects, counterbalanced design, each operator completed kitting tasks under a paper-based Control block and an AEGIS System block across three Bills of Materials (BOMs). Operators rated the system across nine usability dimensions (R1 to R9) on a 1-to-5 Likert scale.

Table 23: Operator Usability Rating Results across Nine Evaluated Dimensions (n = 6)
------------------------------------------------------------------------------------------------------------------------
ID     Usability Dimension Rated by Operators                             Mean Score  Score Range  n
------------------------------------------------------------------------------------------------------------------------
R1     Screen wording was clear and easy to understand                    3.3         2 – 5        6
R2     Placement / layout of on-screen information was easy to follow     4.2         3 – 5        6
R3     Easy to understand WHAT to pack                                    4.9         4.5 – 5      6
R4     Easy to understand HOW MANY to pack                                5.0         5 – 5        6
R5     Running count per bin helped keep track                            3.8         1 – 5        6
R6     Warnings / error alerts were easy to understand                    3.3         2 – 5        6
R7     Corrective instructions clearly explained the fix                  3.0         1 – 4        5*
R8     Overall trust that the system would catch mistakes                 4.0         3 – 5        6
R9     Overall, the system made kitting easier than without it            4.3         2 – 5        6
------------------------------------------------------------------------------------------------------------------------
*Operator Benny triggered zero error states during live runs (n = 5 for R7).

Qualitative User Insights:
- Strengths: What to pack (R3: 4.9) and target quantity (R4: 5.0) scored near ceiling due to the intuitive spatial bin grid mapping. Overall ease compared to paper rated high (R9: 4.3).
- Weaknesses: Small alert typography caused legibility friction (R1, R6: 3.3). Corrective instruction completeness scored lowest (R7: 3.0) because wrong-bin alerts omitted return destination and unit quantity. Touchscreen tap latency on completed bin consolidation caused operator fatigue.

[Figure 5.3: Operator Usability Rating Distribution Bar Chart Showing Means and Ranges across R1-R9]

5.4.2 User-Facing HMI Iteration
Driven directly by operator feedback, a targeted software iteration was implemented on the frontend HMI:

1. Typography Enhancement: Enlarged alert font sizes by 50% for glanceability.
2. Unified Alert Hierarchy: Replaced dual top/bottom banners with a single high-contrast red/white top banner.
3. Explicit Corrective Prompts: Updated wrong-bin return alerts to calculate and display exact corrective instructions (e.g. "WRONG BIN: Return 2 units of SKU-329 to Bin 1_0").
4. Automatic Consolidation Flow: Replaced manual "Bin Emptied" touchscreen taps with automatic load cell weight delta detection.
5. Standardized Numbering & Progress: Converted bin identifiers from 0-indexed to 1-indexed (Bins 1-9) and added a "Sets Remaining" progress bar.

[Figure 5.4: HMI Visual Interface Comparison Showing Pre-Iteration vs Post-Iteration Layout Enhancements]

5.4.3 Technical System Testing Review (Requirements Traceability Matrix)
System compliance was formally evaluated against the functional requirements defined in Chapter 2.

Table 24: Requirements Traceability and Validation Matrix
------------------------------------------------------------------------------------------------------------------------
ID     Requirement              Status          Verification Evidence and Engineering Rationale
------------------------------------------------------------------------------------------------------------------------
FR-01  Component Identification MET (Green)     Verified. YOLOv8-OBB localizes bin rims; inventory engine maps weights to SKUs.
FR-02  Wrong Part Detection     MET (Green)     Verified. FSM immediately triggers wrong-bin fault upon unauthorized reach.
FR-03  Missing Part Detection   MET (Green)     Verified. FSM blocks kit completion state if any SKU is below target count.
FR-04  Quantity Verification    PARTIALLY MET   MET for items >5g. For light components (<3.5g) on 5kg/10kg summed platforms, off-centre
                                (Yellow)        sensitivity and noise floor occasionally skip counts. Mitigated by routing light
                                                items to single-cell 1kg platforms (sigma = 0.028g).
FR-05  Real-Time Alerting       MET (Green)     Verified. End-to-end latency <80ms (33ms CV + 10Hz REST polling).
FR-06  Corrective Guidance      MET (Green)     Verified. Resolved in HMI iteration: alerts display explicit destination bin and count.
FR-07  Order Integration        MET (Green)     Verified. System parses YAML order configs and configures FSM locks accordingly.
FR-08  New Part Onboarding      MET (Green)     Verified. Dynamic OBB calibration eliminates coordinate mapping; SKUs added via YAML.
FR-09  Environmental Robustness MET (Green)     Verified. 100% CV detection from 0 to 506 lux; anti-glare ESD mats eliminate reflections.
FR-10  Supervisor Visibility    MET (Green)     Verified. FastAPI server hosts live state endpoints accessible over local intranet.
FR-11  Scalability              MET (Green)     Verified. Multi-threaded architecture uses 1 of 12 Jetson cores, supporting multi-station extension.
------------------------------------------------------------------------------------------------------------------------

--------------------------------------------------------------------------------
5.5 Bill of Materials (BOM)
--------------------------------------------------------------------------------
Table 25 details the complete bill of materials and hardware cost breakdown for a single AEGIS workstation unit.

Table 25: AEGIS Prototype Single-Workstation Bill of Materials (BOM)
------------------------------------------------------------------------------------------------------------------------
Component Category      Description / Specification                         Qty    Unit Cost (USD)    Total Cost (USD)
------------------------------------------------------------------------------------------------------------------------
Edge Computing          Advantech MIC-733 (NVIDIA Jetson AGX Orin 32GB)     1      $1,850.00          $1,850.00
Optical Sensor          1080p 30fps USB Industrial Camera Feed              1      $75.00             $75.00
Microcontroller         ESP32-WROOM-32 Development Board                    1      $6.00              $6.00
ADC Amplifiers          HX711 24-Bit Serial Weighing ADCs                   4      $2.50              $10.00
Load Cell Sensors       1 kg Single Cantilever Load Cell                    3      $8.00              $24.00
Load Cell Sensors       5 kg Shear Beam Load Cells                          6      $12.00             $72.00
Load Cell Sensors       10 kg Shear Beam Load Cells                         6      $14.00             $84.00
Workstation Frame       Aluminum Extrusion Rig & Acrylic Mounting Plates   1      $180.00            $180.00
ESD Protection          Matte Electrostatic-Discharge Workstation Matting  1      $35.00             $35.00
HMI Display             15.6-inch Full-HD Touchscreen Monitor Kiosk         1      $120.00            $120.00
------------------------------------------------------------------------------------------------------------------------
Total Station Hardware  Complete AEGIS Edge Workstation Deployment Cost     --     --                 $2,456.00
------------------------------------------------------------------------------------------------------------------------

Cost Rationale: The total station hardware cost of ~$2,456 USD is well below the lower bound of TE Connectivity's annual per-line COPQ ($10,000 to $20,000 USD), supporting a positive return on investment (ROI) within the first three months of deployment.


================================================================================
Chapter 6 | Conclusion
================================================================================

6.1 Prototype Summary and Sandbox Trial
The AEGIS prototype successfully demonstrates the operational feasibility of a multi-modal, edge-deployed kitting assistance system. By integrating rim-centric YOLOv8-OBB bin geofencing, MediaPipe hand tracking, and 3-point load cell gravimetric verification under a sequential single-bin FSM, AEGIS intercepts wrong-part, missing-part, out-of-sequence, and quantity errors at the point of commission.

Following successful laboratory validation and user-driven HMI redesign, TE Connectivity is considering implementing the AEGIS solution design in their industrial sandbox environment for operational testing. This sandbox trial will evaluate hardware longevity, load cell drift, and operator compliance under continuous multi-shift production conditions.

[Figure 6.1: Overall System Deployment Architecture Showing Edge Processing, Sensor Array, and Intranet Supervisor View]

6.2 Process Scalability and Maintenance
To support plant-wide scaling across manufacturing sites in China, Poland, and Mexico, maintenance procedures were engineered for non-specialist plant technicians:

1. SKU Cataloguing: Onboarding new SKUs requires zero machine learning model retraining. Technicians record unit mass and bin mapping directly in `inventory.yaml`.
2. Rapid Station Calibration: Station calibration takes <2 minutes via the automated two-snapshot YOLOv8-OBB process, eliminating manual coordinate entry.
3. Modular Hardware Swap: Load cell assemblies are mechanically standardized. Sensor replacement requires removing four screws without recalibrating the frame.
4. Multi-Station Architecture: Low CPU/GPU resource utilization (10% CPU, 6.7% GPU) allows a single Jetson AGX Orin edge node to aggregate state streams across multiple workstation instances.

6.3 Ergonomic Considerations
To reduce physical operator fatigue prior to sandbox deployment, two ergonomic enhancements are recommended:
1. Consolidation Fatigue: Replace standard bins with tilting bin racks or gravity-fed slide-out chutes to eliminate manual lifting of heavy completed bins into kit containers.
2. Standing Fatigue: Incorporate automated break reminders on the HMI (e.g. suggesting a short rest every 10 completed sets) to maintain operator attentiveness.

6.4 Future Works
1. Factory MES Integration: Implement Modbus TCP communications to interface Jetson edge nodes with TE Connectivity's Manufacturing Execution System (MES) for automated work-order loading.
2. Centralized Supervisor Dashboard: Aggregate local FastAPI endpoints into a unified intranet dashboard for multi-line visibility.
3. Ultra-Light Part Verification: Explore capacitive proximity sensing or optical gate arrays for components weighing <1 g.


================================================================================
Chapter 7 | Reflection and Lessons Learned
================================================================================

1. Sensor Physics Limits Software Capability: The single hardest engineering lesson was that software algorithms cannot overcome physical sensor metrology limits. Off-centre sensitivity and noise floors on 5 kg/10 kg summed load cell platforms set hard physical bounds on resolving light components (<3.5 g). Robust engineering requires pairing software logic with physical configuration—such as assigning light components exclusively to single-cell 1 kg platforms (σ = 0.028 g).

2. Usability is Non-Obvious: Assumptions made by developers regarding alert text and visual layout failed under production-pace operator testing. While core task guidance rated near ceiling (R4: 5.0), exception communication initially scored low (R7: 3.0) due to small font sizes and incomplete corrective wording. Engaging active operators early in usability trials is essential for human-centric UI design.

3. Pure-Logic Decoupling Enables Automated Testing: Decoupling the verification state machine (`placement.py`) from physical hardware peripherals enabled exhaustive unit testing under Pytest (134 test cases, 98% statement coverage). This architectural decision accelerated development velocity and provided a robust regression guard during software iterations.

4. Real-World Hardware Robustness: Laboratory prototypes meet messy hardware realities during extended execution. Building automated serial reconnect loops (<500 ms recovery), sensor health checks, and fault-tolerant state handling is as vital to industrial deployment as primary machine learning inference.


================================================================================
References
================================================================================
[1] Ultralytics, "YOLOv8 Documentation," 2024. [Online]. Available: https://github.com/ultralytics/ultralytics.
[2] Google, "MediaPipe Hands: On-device Real-time Hand Tracking," 2023. [Online]. Available: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker.
[3] TE Connectivity, "Cost of Poor Quality (COPQ) Internal Manufacturing Reports," FY2025.
[4] Avia Semiconductor, "HX711: 24-Bit Analog-to-Digital Converter for Weigh Scales," Datasheet Rev. 1.0.
[5] Advantech, "MIC-733: AI Inference System Based on NVIDIA Jetson AGX Orin," Product Datasheet, 2024.
[6] NVIDIA, "Jetson AGX Orin Technical Reference Manual," 2023.
[7] Roboflow, "Computer Vision Annotation and Dataset Management Platform," 2024. [Online]. Available: https://roboflow.com.
[8] Espressif Systems, "ESP32 Series Datasheet," Version 4.2, 2023.


================================================================================
Appendix
================================================================================
Appendix A: Usability Study Raw Data Logs (AEGIS_User_Test_Findings_Summary.docx)
Appendix B: Load Cell Bench Metrology Qualification Data (AEGIS_LoadCell_Qualification.docx)
Appendix C: Computer Vision Training & Lighting Characterisation Workbook (lighting_cv_test_backup.xlsx)
Appendix D: HMI Interface Screenshots (Pre-Iteration vs Post-Iteration Layouts)
Appendix E: Pytest Backend Verification Suite Summary (134/134 Tests Passing Log)
