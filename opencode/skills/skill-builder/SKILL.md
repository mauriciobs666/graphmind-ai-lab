---
name: skill-builder
description: Creates new OpenCode skills from user requirements. Use to build SKILL.md files with proper structure, conventions, and best practices for specialized task assistance.
license: Apache-2.0
compatibility: opencode
metadata:
  audience: developers
  workflow: skill-creation
---

## What I do

- Receive requirements describing a specialized skill to create
- Generate SKILL.md files following OpenCode skill conventions
- Define skill scope, use cases, and prerequisites
- Include code examples, best practices, and common issues
- Create supporting files when needed (templates, scripts, resources)

## When to use me

Use this when:
- User wants to create a new specialized skill for OpenCode
- User provides requirements for a domain-specific assistant
- User describes actions or workflows to be encapsulated as a skill
- User wants to formalize expertise into a reusable skill

## Prerequisites

- Understand the user's requirements and target audience
- Identify the skill's scope and boundaries
- Determine required tools, languages, or conventions

## Discovery Paths

OpenCode searches these locations for skills:

- Project config: `.opencode/skills/<name>/SKILL.md`
- Global config: `~/.config/opencode/skills/<name>/SKILL.md`
- Project Claude-compatible: `.claude/skills/<name>/SKILL.md`
- Global Claude-compatible: `~/.claude/skills/<name>/SKILL.md`
- Project agent-compatible: `.agents/skills/<name>/SKILL.md`
- Global agent-compatible: `~/.agents/skills/<name>/SKILL.md`

## Skill Structure

### SKILL.md Frontmatter

Only these frontmatter fields are recognized:

```yaml
---
name: <skill-name>        # Required
description: <text>       # Required (1-1024 chars)
license: <license>        # Optional (e.g., Apache-2.0, MIT)
compatibility: opencode   # Optional
metadata:                 # Optional (string-to-string map)
  key: value
---
```

Unknown fields are ignored.

### Name Validation Rules

The `name` field must:
- Be 1-64 characters
- Use lowercase alphanumeric with single hyphen separators
- Not start or end with `-`
- Not contain consecutive `--`
- Match the directory name

Valid pattern: `^[a-z0-9]+(-[a-z0-9]+)*$`

### SKILL.md Sections

1. **What I do** - Core capabilities (bullet list)
2. **When to use me** - Trigger conditions with "Use this when:" bullets
3. **Prerequisites** - Required tools, knowledge, or setup
4. **Conventions** - Domain-specific standards and patterns
5. **Best Practices** - Recommended approaches with examples
6. **Common Issues** - Frequent problems and solutions

## Creation Process

### Step 1: Analyze Requirements
- Extract skill name (slug format: `kebab-case`)
- Identify target audience and use cases
- Determine scope (what to include/exclude)

### Step 2: Draft Frontmatter
- Name: derived from requirements or user input
- Description: concise 1-2 sentences (max 1024 characters)
- Metadata: appropriate audience and workflow tags

### Step 3: Write Core Sections
- What I do: 5-10 specific capabilities
- When to use me: 5-10 trigger scenarios
- Prerequisites: tools, knowledge, setup requirements

### Step 4: Add Domain Content
- Conventions: file structure, naming, style guidelines
- Best Practices: annotated code examples
- Common Issues: troubleshooting guide

### Step 5: Review and Finalize
- Ensure consistency with OpenCode conventions
- Verify all examples are accurate
- Check for completeness and clarity

## Output Files

- `SKILL.md` - Main skill definition (single file per skill)
- Supporting files in the same directory: templates, scripts, configs

**Note:** OpenCode only loads one `SKILL.md` per skill folder. Supporting files must be referenced within the skill content.

## Complex Workflows

For multi-phase workflows, organize SKILL.md with clear sequential sections:

1. **Overview** - High-level workflow description
2. **Phase 1: [Name]** - First step with prerequisites and actions
3. **Phase 2: [Name]** - Subsequent steps
4. **Decision Points** - Branch logic and condition handling
5. **Completion** - Finalization and verification steps

### Workflow Example Structure
```markdown
## Workflow Overview

Multi-step process for [goal].

## Phase 1: Setup

Prerequisites and initial configuration...

## Phase 2: Execution

Core implementation steps...

## Phase 3: Validation

Testing and verification...
```

### Referencing Supporting Files

Include templates or scripts as separate files in the skill directory and reference them:
```markdown
See `@skill-dir/template.sh` for the deployment script template.
```

## Directory Structure for Complex Skills

```
~/.config/opencode/skills/<skill-name>/
├── SKILL.md           # Main definition (required)
├── template.sh       # Optional: script templates
├── config.yaml       # Optional: configuration examples
└── README.md         # Optional: additional documentation
```

## Permission Configuration

Control skill access via `opencode.json`:

```json
{
  "permission": {
    "skill": {
      "*": "allow",
      "pr-review": "allow",
      "internal-*": "deny",
      "experimental-*": "ask"
    }
  }
}
```

| Permission | Behavior |
|------------|----------|
| `allow` | Skill loads immediately |
| `deny` | Skill hidden from agent |
| `ask` | User prompted before loading |

## Quality Checklist

- [ ] Name matches directory name and follows validation rules
- [ ] Description is 1-1024 characters
- [ ] Frontmatter uses only recognized fields
- [ ] All required sections have meaningful content
- [ ] Code examples are syntactically correct
- [ ] Prerequisites are realistic and achievable
- [ ] Common issues address real problems
- [ ] Content follows markdown best practices

## Troubleshooting

If a skill does not load:
1. Verify `SKILL.md` is spelled in all caps
2. Check that frontmatter includes `name` and `description`
3. Ensure skill names are unique across all locations
4. Check permissions in `opencode.json`

## Example Output

```markdown
---
name: docker-helper
description: Assists with Docker container management, Dockerfile creation, and containerized application workflows.
license: Apache-2.0
compatibility: opencode
metadata:
  audience: developers
  workflow: devops
---

## What I do

- Create optimized Dockerfiles for various languages
- Build and manage Docker images and containers
- Set up multi-stage builds for smaller images
- Configure container networking and volumes
- Debug containerized applications

## When to use me

Use this when:
- Writing Dockerfiles from scratch
- Optimizing existing Dockerfiles
- Troubleshooting container issues
- Setting up development environments with Docker
```

## Tips

- Keep descriptions concise (1-1024 characters max)
- Use bullet points for scannability
- Include real-world code examples
- Group related information logically
- Cross-reference related skills when applicable
- For very long skills, use horizontal rules (`---`) to separate major sections
- Use numbered lists for sequential steps in workflows
