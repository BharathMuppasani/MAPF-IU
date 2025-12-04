
import cpp_collision
try:
    plan = cpp_collision.YieldPlan()
    print(f"Has rejected_candidates: {hasattr(plan, 'rejected_candidates')}")
    print(f"Value: {plan.rejected_candidates}")
except Exception as e:
    print(f"Error: {e}")
