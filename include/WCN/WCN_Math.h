#ifndef WCN_MATH_H
#define WCN_MATH_H

// ============================================================================
// WCN_Math.h — the single entry point.
//
// This header used to dispatch on WCN_USE_DST_MODE between two complete API
// declarations: a destination-out-param API (WCN_MATH_DST.h) and a
// return-by-value API declared inline below. The two libraries implemented the
// same math under the same symbol names but with different calling
// conventions, so a build that produced one and declared the other linked
// cleanly and then corrupted the stack at run time.
//
// Only the destination-out-param API is maintained. This header is now a thin
// alias for it; include it and you get WCN_MATH_DST.h.
//
// Migration (return-by-value -> destination-out-param):
//   before:  WCN_Math_Vec2 v = wcn_math_Vec2_add(a, b);
//   after:   WCN_Math_Vec2 v; wcn_math_Vec2_add(&v, a, b);
//
// Scalar-returning functions (dot/length/angle/equals/...) are unchanged.
// The `set` family drops its first argument:
//   before:  WCN_Math_Vec2 v = wcn_math_Vec2_set(old, x, y);
//   after:   wcn_math_Vec2_set(&v, x, y);
// ============================================================================

#include "WCN/WCN_MATH_DST.h"

#endif // WCN_MATH_H