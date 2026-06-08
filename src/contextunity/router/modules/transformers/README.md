# Ingestion Transformers (Migration Plan)

**STATUS: WORK IN PROGRESS / PRE-MIGRATION**

This directory contains logic components (`ner.py`, `keyphrases.py`, `taxonomy.py`, `ontology.py`, etc.) designated for the upcoming **Ingestion Migration to Router** plan.

## Purpose

The architectural plan is to route ingestion requests through the `contextunity.router` before they reach the `contextunity.worker` or `contextunity.brain`. 
The `router` acts as the orchestrator to decide *how* processing components (like semantic splitters, Named Entity Recognition, and Knowledge Graph extraction here) are applied based on the project's payload, while delegating the heavy execution load to the worker plane via Temporal workflows.

## Do Not Delete

Even if these modules are not actively imported by LangGraph compiler modules (`cortex/compiler/...`) *yet*, **this code is active infrastructure for the ingestion pipeline roadmap** and must be preserved.

*Note: As part of the upcoming **flat_memory** plan, certain sub-components of this pipeline may eventually be moved directly to `contextunity.brain` while maintaining the Router's orchestrational role.*
