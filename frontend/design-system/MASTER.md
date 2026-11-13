# Calo Khay — UI design system

Applied after an audit with `ui-ux-pro-max` on 2026-11-12.

## Direction

Minimal nutrition workspace. Preserve the user's request for concise Vietnamese copy, no slogans, decorative steppers or parenthetical explanations. Dark surfaces with a soft green action color; Inter supports Vietnamese and a clear numeric hierarchy.

The skill's generated recommendation was Dark Mode + Inter + green CTA. Its generic single-column landing-page structure does not fit an image analysis workspace, so use two columns above 900 px and one column below. Retain the existing vanilla JS/Vite architecture and CSS variables; Tailwind examples are implementation guidance, not a reason to migrate the stack. No light mode is currently offered.

## Tokens

- Canvas `#141615`, navigation `#111412`, surface `#1d211e`, border `#333a34`.
- Text `#eceeed`, secondary `#adb7af`, action `#d0edac` with dark text.
- Summary surface `#242d22`; confirmation surface `#29251e` with amber text.
- Typography: heading 28, section/body 14–16, metadata 12, main number 54 px.
- Spacing: 4 / 8 / 12 / 16 / 24 / 32. Card radius 14, buttons 8.
- Interactive color transitions: 180 ms; reduced-motion override required.

## Interaction

- Image and SVG must share identical bounds and source aspect ratio.
- Numbered markers match numbered food rows; full food names remain accessible in SVG titles and aria labels.
- Original / detected-region view controls use `aria-pressed`.
- Color is supplemented with numbers, text status and visible selection.
- Main controls and mobile actions have at least 44 px targets. Body copy on mobile is 16 px; compact data labels are 12–14 px.
- Dialog titles are associated using `aria-labelledby`; navigation remains named when icon-only.
- Keep a single dominant save action, within the unified result card.
- Help, data notes and sources use progressive disclosure.

## Audit findings addressed

1. Secondary text was 10–11 px: raise type scale and contrast.
2. Image labels overlapped: replace them with numbered markers plus original-image toggle.
3. Result summary, confirmation and list felt disconnected: unify them in one card.
4. Desktop delete/portion controls were too small: increase hit areas; retain compact icons.
5. Hidden navigation text lost accessible names: add explicit labels.
6. Dialog titles were unassociated: add accessible title references.
7. Hover feedback was uneven: add stable 180 ms color/border transitions.

## Verification scope

Check image/overlay geometry, horizontal overflow and visible controls at 375, 768, 1024 and 1440 px. Exercise image view toggle, item editing, confirmation, search and navigation. Existing domain tests cover arithmetic and API error mapping. Browser checks do not replace a full screen-reader or cross-browser audit.

### Verified in the local browser

- 375 / 768 / 1024 / 1440 px: no horizontal overflow; image and SVG bounds match.
- Main actions, delete controls and portion choices meet 44 px dimensions in the inspected scan view.
- Switching to the original image removes the overlay; switching back restores it.
- Selecting marker 2 highlights its food row and moves keyboard focus to that row.
- Changing potato weight to 100 g, confirming a food, and searching `ca vien` work.
- Core palette contrast: text 13.98:1, secondary text 7.89:1, primary button 12.17:1, summary 12.73:1, warning 8.29:1, navigation 7.26:1. These are token-pair checks, not an exhaustive automated accessibility audit.
- Vite build and all 7 existing domain tests pass.
