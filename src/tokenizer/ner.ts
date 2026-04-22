/**
 * Stage 4 — Named Entity Recognition
 * Uses compromise.js to detect entities that should NOT be decomposed.
 * Named entities emit LIT: tokens directly.
 */

// @ts-ignore compromise has no typed exports for this pattern
import nlp from 'compromise'

export interface EntityMap {
  /** Maps lowercase surface form → true for detected entities */
  entities: Set<string>
}

export function detectEntities(text: string): EntityMap {
  const doc = nlp(text)
  const entities = new Set<string>()

  // People
  doc.people().forEach((m: any) => {
    const t = m.text().toLowerCase()
    if (t) entities.add(t)
  })

  // Places
  doc.places().forEach((m: any) => {
    const t = m.text().toLowerCase()
    if (t) entities.add(t)
  })

  // Organizations
  doc.organizations().forEach((m: any) => {
    const t = m.text().toLowerCase()
    if (t) entities.add(t)
  })

  return { entities }
}

export function isEntity(word: string, entityMap: EntityMap): boolean {
  return entityMap.entities.has(word.toLowerCase())
}
