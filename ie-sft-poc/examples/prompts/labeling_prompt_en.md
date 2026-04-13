# Information Extraction Labeling Guidelines

## Overview

This guide provides instructions for annotators performing information extraction tasks on English text.

This project supports three types of information extraction tasks:
- **Key-Value (KV) Extraction**: Extract specific fields and their values from text
- **Named Entity Recognition (NER)**: Identify and classify entities like persons, organizations, locations
- **Relation Extraction (RE)**: Extract semantic relationships between entities

---

## 1. Key-Value (KV) Extraction Task

### 1.1 Task Description

Key-value extraction involves identifying and extracting structured information from documents such as invoices, receipts, purchase orders, and product listings.

**Example:**
```
Input: "Invoice #INV-2024-001 issued by TechCorp Inc. on March 10, 2024. Total: $1,700. Status: Paid."

Output:
- invoice_number: "INV-2024-001"
- company_name: "TechCorp Inc."
- issue_date: "March 10, 2024"
- total_amount: "$1,700"
- status: "Paid"
```

### 1.2 Extraction Rules

1. **Adhere to Field Definitions**: Extract only the fields specified in the schema.
2. **Copy Text Exactly**: Extract the exact text as it appears in the document.
3. **Include Whitespace**: If a value contains spaces, include them as written.
4. **Preserve Special Characters**: Include currency symbols, punctuation, and formatting.
5. **Handle Missing Values**: Use `null` or empty string if information is absent.

### 1.3 KV Extraction Examples

**Example 1: Product Information**
```
Input: "Product: Premium Wireless Earbuds | SKU: PW-2024-001 | Price: $299.99 | Stock: 45 units"

Fields:
- product_name: "Premium Wireless Earbuds"
- sku: "PW-2024-001"
- price: "$299.99"
- stock_quantity: "45 units"
```

**Example 2: Order Information**
```
Input: "Order #ORD-2024-567890 placed on March 14, 2024 at 2:30 PM. Shipping address: New York, NY. Cost: Free shipping."

Fields:
- order_id: "ORD-2024-567890"
- order_datetime: "March 14, 2024 at 2:30 PM"
- delivery_address: "New York, NY"
- shipping_fee: "Free shipping"
```

---

## 2. Named Entity Recognition (NER) Task

### 2.1 Task Description

Named Entity Recognition involves identifying and classifying named entities in text. These include persons, organizations, locations, dates, products, and monetary amounts.

### 2.2 Entity Type Definitions

| Type | Description | Examples |
|------|-------------|----------|
| PERSON | Names of individuals | John Smith, Steve Jobs, Elon Musk |
| ORG | Names of organizations/companies | Apple Inc., Google, Microsoft Corporation |
| LOC | Geographic locations | New York, United States, Mountain View |
| DATE | Temporal expressions | March 15, 2024, yesterday, next Monday |
| PRODUCT | Product/service names | iPhone 15, Windows 11, Gmail |
| MONEY | Monetary amounts | $1,000, 100 euros, $2.5 million |

### 2.3 Entity Extraction Rules

1. **Exact Boundaries**: Mark the exact start and end positions of each entity.
2. **Extract All Occurrences**: If an entity appears multiple times, extract all instances.
3. **Complete Entities Only**: Extract the full entity, excluding unnecessary modifiers.
   - Incorrect: "CEO John Smith" → extract "CEO" (X)
   - Correct: "CEO John Smith" → extract "John Smith" (O)
4. **Type Accuracy**: Choose the most specific and appropriate entity type.
5. **No Overlapping Entities**: Each text span should be labeled with only one type.

### 2.4 Entity Extraction Examples

**Example:**
```
Input: "Google was founded by Larry Page and Sergey Brin in 1998 at Stanford University in California."

Extracted Entities:
- "Google" → ORG
- "Larry Page" → PERSON
- "Sergey Brin" → PERSON
- "1998" → DATE
- "Stanford University" → ORG
- "California" → LOC
```

---

## 3. Relation Extraction Task

### 3.1 Task Description

Relation extraction identifies and classifies semantic relationships between identified entities in text.

### 3.2 Relation Type Definitions

| Relation Type | Description | Example |
|---------------|-------------|---------|
| works_for | Person employed by organization | "John works at Google" |
| located_in | Entity located in a place | "Apple's headquarters is in Cupertino" |
| founded_by | Organization founded by person | "Google was founded by Larry Page" |
| owns | Person/org owns entity | "Elon Musk owns Tesla" |
| partner_with | Organizations in partnership | "Samsung partners with Google" |
| headquartered_in | Org headquarters location | "Microsoft is headquartered in Seattle" |

### 3.3 Relation Extraction Rules

1. **Clear Head-Tail Structure**: Distinguish between the subject (head) and object (tail) of the relation.
2. **Context-Based Inference**: Extract relations that can be inferred from context, not just explicit mentions.
3. **Multiple Relations**: If multiple relations exist, extract all of them.
4. **Correct Directionality**: Ensure the relation direction is correct (subject → relation → object).
5. **Type Accuracy**: Select the most specific and appropriate relation type.

### 3.4 Relation Extraction Examples

**Example:**
```
Input: "Tim Cook serves as the CEO of Apple Inc., which is headquartered in Cupertino, California."

Extracted Relations:
1. Head: "Tim Cook" (PERSON) → Relation: works_for → Tail: "Apple Inc." (ORG)
2. Head: "Apple Inc." (ORG) → Relation: headquartered_in → Tail: "Cupertino, California" (LOC)
```

---

## 4. Common Mistakes and What to Avoid

### Mistake 1: Entity Boundary Errors
```
✗ Incorrect: "Apple Inc" (missing period)
✓ Correct: "Apple Inc." (complete name)
```

### Mistake 2: Relation Direction Confusion
```
✗ Incorrect: "Tim Cook" → located_in → "Apple Inc."
✓ Correct: "Apple Inc." → located_in → "Cupertino, California"
```

### Mistake 3: Over-Classification
```
✗ Incorrect: "new smartphone" → PRODUCT (too broad)
✓ Correct: "iPhone 15" → PRODUCT (specific)
```

### Mistake 4: Adding Implicit Information
```
✗ Incorrect: Inferring information not explicitly stated in text
✓ Correct: Extract only information explicitly present in the document
```

### Mistake 5: Incomplete Entity Names
```
✗ Incorrect: "United States" when text says "United States of America"
✓ Correct: "United States of America" (complete name as written)
```

---

## 5. Format Specifications

### Input Format
- Plain text in English
- May contain special characters and formatting
- Context surrounding key information should be preserved

### Output Format

**KV Extraction:**
```json
{
  "field_name": "extracted_value",
  "field_name2": "extracted_value2"
}
```

**NER:**
```json
[
  {"text": "entity text", "type": "PERSON", "start": 0, "end": 15},
  {"text": "another entity", "type": "ORG", "start": 20, "end": 34}
]
```

**Relation:**
```json
[
  {
    "head": "Tim Cook",
    "head_type": "PERSON",
    "relation": "works_for",
    "tail": "Apple Inc.",
    "tail_type": "ORG"
  }
]
```

---

## 6. Quality Assurance Checklist

Before finalizing your annotations, verify:

- [ ] All specified fields are extracted for KV tasks?
- [ ] All entities are identified and correctly classified?
- [ ] All evident relations are extracted?
- [ ] Entity boundaries are precise (no missing or extra characters)?
- [ ] Relation direction is correct (head → tail)?
- [ ] No information is added that isn't in the text?
- [ ] Consistent annotation style throughout the document?
- [ ] Proper JSON formatting (if applicable)?
- [ ] No duplicate extractions?

---

## 7. Special Cases

### Abbreviated Names
```
Input: "Dr. Jane Smith from the WHO"
Annotations:
- "Jane Smith" → PERSON
- "WHO" → ORG (World Health Organization - extract acronym as written)
```

### Compound Entities
```
Input: "New York-based Google subsidiary"
Annotations:
- "Google" → ORG (the primary entity, not the subsidiary)
- "New York" → LOC
```

### Monetary Amounts with Ranges
```
Input: "The company reported revenue between $50 million and $100 million"
Annotations:
- "$50 million" → MONEY
- "$100 million" → MONEY
```

---

## 8. Resources and Support

- Schema definitions: See `schema.py` in project documentation
- Validation script: See `validate_canonical_dataset.py`
- Example datasets: See `examples/canonical_samples/`

For questions or ambiguous cases, please contact your team lead or project manager.

---

## 9. Quick Reference

**Key Principles:**
1. Be precise: Exact text, exact boundaries
2. Be consistent: Same entities labeled same way throughout
3. Be complete: Extract all instances of requested types
4. Be accurate: Correct types and relation directions
5. Be honest: Don't add information not in the text
