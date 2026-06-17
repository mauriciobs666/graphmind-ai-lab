---
name: write-tutorial
description: Creates structured learning paths and generates comprehensive markdown tutorials with summaries and chapters. Uses comparison-driver for option analysis when needed.
license: Apache-2.0
compatibility: opencode
metadata:
  audience: developers
  workflow: content-creation
---

## Persona & Tone

- Adopt an **encouraging, patient, and pedagogical** tone.
- Act as an experienced mentor guiding a learner via clear, practical examples.
- Avoid dense academic jargon unless it is defined immediately.

## What I do

- Analyze user intent to identify the core topic and learning objectives
- Systematically create a learning path with approximately 7 items
- Generate a comprehensive summary markdown file covering the topic overview
- Create individual markdown files for each chapter in the learning path
- Use comparison-driver skill to evaluate and select the best approach when multiple options exist
- Structure content for progressive learning (beginner to advanced)
- Include practical examples and exercises in each chapter

## When to use me

Use this when:
- User wants to create a tutorial on a specific topic
- User needs a structured learning path for a subject
- User asks for educational content with progressive complexity
- User wants to document their expertise in a tutorial format
- User needs a quick start guide for a technology or concept

## Prerequisites

- Clear intent or topic description from the user
- Understanding of the target audience's skill level
- Access to comparison-driver skill for option analysis
- File write capabilities to create markdown files

## Conventions

### Output Structure & File Naming

- **Output Location:** All files must be generated inside a dedicated directory: `tutorials/<topic-slug>/`
- Summary file: `SUMMARY.md`
- Chapter files: `01-chapter-name.md`, `02-chapter-name.md`, etc.
- Use kebab-case for chapter names, title case for display titles

### Learning Path Structure

A learning path should ideally progress through 7 distinct stages:
1. Introduction/Overview (what and why)
2. Core concepts/Fundamentals
3. Basic usage/Getting started
4. Intermediate concepts
5. Advanced topics
6. Practical examples
7. Capstone Project (combines concepts from chapters 1-6 into a single functional project)

### Content Organization & Aesthetics

Each chapter must include:
- **Clear learning objective** at the beginning.
- **Visuals & Callouts**: Use GitHub-flavored alerts (`> [!NOTE]`, `> [!TIP]`, `> [!WARNING]`, `> [!IMPORTANT]`) and Mermaid diagrams (```mermaid```) to break up text and illustrate architectures.
- **Rich Content**: Conceptual explanations coupled with practical code examples.
- **Engagement**: A "Check Your Knowledge" section at the end (short quiz, questions, or mini-challenge) to actively reinforce learning.

The `SUMMARY.md` file MUST begin with YAML frontmatter containing metadata:
```yaml
---
difficulty: Beginner | Intermediate | Advanced
estimated_time: [e.g., 2 hours]
prerequisites: [list of prerequisites]
---
```

## Best Practices

### Creating a Learning Path

```
1. Analyze the intent:
   - What is the topic?
   - What is the target audience?
   - What is the expected outcome?

2. Break down into 7 progressive steps:
   - Start with foundations
   - Build toward advanced concepts
   - Include practical application

3. Use comparison-driver when:
   - Multiple tools/frameworks exist for the same purpose
   - Different approaches have trade-offs
   - Audience needs guidance on what to choose
   - *Requirement:* If used, explicitly include a "Technology Choices" section in Chapter 1 summarizing the trade-offs and justifying the chosen tools.
```

### Example Learning Path Creation

For intent: "Learn Python web development"
- Chapter 1: Introduction to Web Development with Python
- Chapter 2: Setting Up Your Development Environment
- Chapter 3: Understanding HTTP and Web Frameworks
- Chapter 4: Building Your First Web Application
- Chapter 5: Working with Databases
- Chapter 6: Authentication and Security
- Chapter 7: Deployment and Best Practices

## Common Issues

### Vague Intent

**Problem:** User provides insufficient detail about what they want to learn.
**Solution:** Ask clarifying questions:
- What is your current skill level?
- What do you want to achieve after completing the tutorial?
- Do you have any time constraints?

### Overlapping Content

**Problem:** Chapters have redundant information.
**Solution:** Ensure each chapter:
- Has a distinct focus
- Builds on previous chapters
- Contains unique, non-repetitive content

### Missing Prerequisites

**Problem:** Tutorial assumes knowledge the audience doesn't have.
**Solution:** Include a prerequisites section in the summary and clearly state required knowledge in each chapter.

### Too Much Content

**Problem:** Creating overly long chapters that overwhelm learners.
**Solution:** Keep chapters focused:
- Aim for 300-800 words per chapter
- Break complex topics into sub-chapters
- Use clear section headers

## Workflow

### Phase 1: Intent Analysis

1. Receive and clarify the user's intent
2. Identify learning objectives
3. Determine target audience expertise level

### Phase 2: Learning Path Creation

1. Create 7-item structured outline
2. Use comparison-driver if multiple approaches exist
3. Define chapter objectives

### Phase 3: Content Generation

1. Write SUMMARY.md with overview and prerequisites
2. Generate each chapter file sequentially
3. Ensure consistent formatting and tone

### Phase 4: Review

1. Verify learning progression makes sense
2. Check that chapters don't overlap
3. Ensure examples are relevant and accurate