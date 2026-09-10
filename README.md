# AEGIS: AI-Enhanced Guidance and Inspection System

AEGIS is a computer vision and sensor-assisted system designed to reduce errors in manual kitting, the process by which a worker selects and packs a specified set of components into a single kit. The project was developed as the final year capstone for the CDE3301/IS305 module at the National University of Singapore (AY2025/26), in partnership with TE Connectivity and with hardware sponsorship from Advantech.

![AEGIS workstation setup](<final%20report/media/Workstation%20Setup.jpg>)
*The complete AEGIS workstation: overhead camera mount, load-cell bin platforms, operator dashboard, and the AI PC that runs the detection pipeline.*

## The Problem

In high-mix, low-volume manufacturing environments, kitting is performed manually because the variety of parts makes full automation impractical. This reliance on manual work introduces the risk of wrong parts, missing parts, and incorrect quantities. According to TE Connectivity, these errors account for an estimated USD 10,000 to USD 20,000 in losses per production line each year, largely because defects are only discovered downstream, where they are far more costly and difficult to trace back to their source.

## Our Approach

Rather than inspecting a kit after it has been packed, AEGIS observes the packing process as it happens. An overhead camera tracks the operator's hand movements to determine which bin is being accessed, while load cells positioned beneath each bin measure the change in weight for every pick. Together, these two independent signals allow the system to confirm, in real time, whether the correct item and the correct quantity have been taken. When a discrepancy is detected, the operator is alerted immediately through a dashboard interface, before the kit moves further down the line.

This dual-sensor design was a deliberate choice. Rather than requiring the system to visually identify each individual component, which is difficult given how frequently TE Connectivity's parts catalogue changes, AEGIS instead recognizes bins and cross-references their known contents with the weight removed. This keeps the system maintainable by production staff without requiring specialist machine learning expertise for every new part introduced.

## System Overview

The workstation is organized into three layers that work together as a closed loop:

- **Physical layer.** An aluminum-extrusion gantry holds the camera overhead and supports a modular arrangement of load-cell bin platforms and kitting bins.
- **Sensor layer.** A camera tracks the operator's hands, while load cells beneath each bin register mass changes as items are picked.
- **Software layer.** An edge processing engine, running locally on an Advantech MIC-733-AO industrial AI PC (built on an NVIDIA Jetson AGX Orin), combines the two sensor streams and checks them against the kit's bill of materials. Results are pushed to a dashboard on an Advantech FPM-215 touchscreen, giving the operator immediate, visual feedback.

Running all processing locally, rather than in the cloud, was a requirement from TE Connectivity to keep production data on-site and to avoid any dependence on network latency.

## Demonstration

A stitched video recording of the system in operation, combining the camera's point of view, the operator dashboard, and a third-person view of an operator performing a kitting task, is available at [`final report/media/AEGIS Demo - Stitched.mp4`](<final%20report/media/AEGIS%20Demo%20-%20Stitched.mp4>).

## Results

The system was evaluated through both developer-led testing and structured user testing with operators.

- Component identification was shown to be reliable across a range of lighting conditions and bin fill levels, and the backend fault-detection logic passed 134 unit tests covering 98 percent of the core decision-making code.
- In user testing with six operators, AEGIS was rated highly for communicating what to pack and how much to pack, and was generally preferred over a paper-based instruction sheet.
- The same testing also found that AEGIS introduced a consistent cycle-time increase of 35.2 percent across all operators, which did not meet the project's original performance target and is noted as the most significant area for further development.
- In a closing evaluation session, a TE Connectivity stakeholder rated the prototype's readiness for deployment at 80 percent, attributing most of the remaining gap to configuration workflow rather than to detection accuracy or hardware reliability.
- A cost analysis found that the base configuration of the system remains within budget expectations and pays for itself well inside a typical evaluation period.

Taken together, these findings support AEGIS as a proof of concept: the underlying idea, that errors can be caught at the moment they occur rather than after the fact, is validated, while its operating speed and ease of configuration remain the primary items to address before the system would be ready for a live production floor.

## Repository Structure

| Folder | Description |
| --- | --- |
| `final report/` | The final written report submitted for assessment, the demonstration video, supporting source material, and setup photography. |
| `aegis-v2/` | The current version of the bin-tracking and hand-tracking pipeline that powers the live system, along with its computer vision models and integration code. |
| `aegis-cv/` | Earlier computer vision model training and evaluation work. |
| `kitting-error-tracker/` | A standalone tool for tracking kitting errors during testing. |
| `esp32/` | Firmware and PCB design files for the load-cell electronics. |
| `Complete model/` | An earlier consolidated version of the detection and tracking pipeline. |

## Project Team

Developed by Lim Kai Ler Ethan, Aw Shuo Jie, Yap Jia Wei, and Yeo Chen Xian, under the supervision of Mr Chan Tong Leong and Mr Eugene Ee, as part of NUS's Engineering Design and Innovation Centre programme.

## Acknowledgements

We are grateful to TE Connectivity for the problem context and their continued involvement in testing and evaluating the prototype, and to Advantech for sponsoring the MIC-733-AO AI PC and FPM-215 HMI used throughout development. Full acknowledgements are included in the final report.

## Report

The complete final report, covering the problem context, design process, testing results, cost analysis, and conclusions in full, is available at [`final report/IS305 - Final Report - Submitted.pdf`](<final%20report/IS305%20-%20Final%20Report%20-%20Submitted.pdf>).
