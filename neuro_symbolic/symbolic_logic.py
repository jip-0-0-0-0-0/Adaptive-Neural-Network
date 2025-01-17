class SymbolicLogic:
    def __init__(self):
        self.rules = []
        self.log = []

    def add_rule(self, condition, action):
        self.rules.append((condition, action))

    def apply_rules(self, context):
        results = []
        for condition, action in self.rules:
            if condition(context):
                results.append(action(context))
        self.log_results(context, results)
        return results

    def log_results(self, context, results):
        log_entry = {
            "context": context,
            "results": results
        }
        self.log.append(log_entry)

    def evolve_rules(self, feedback):
        for rule, action in feedback:
            if rule not in [r[0] for r in self.rules]:
                self.rules.append((rule, action))
        print("Rules evolved based on feedback.")

    def export_rules(self, file_path):
        with open(file_path, "w") as f:
            f.write("Rules:\n")
            for condition, action in self.rules:
                f.write(f"- Condition: {condition.__name__}, Action: {action.__name__}\n")

if __name__ == "__main__":
    logic_system = SymbolicLogic()

    def is_even(context):
        return context % 2 == 0

    def double_number(context):
        return context * 2

    def is_odd(context):
        return context % 2 != 0

    def triple_number(context):
        return context * 3

    logic_system.add_rule(is_even, double_number)
    logic_system.add_rule(is_odd, triple_number)

    test_context = 5
    results = logic_system.apply_rules(test_context)
    print("Input Context:", test_context)
    print("Results:", results)

    logic_system.evolve_rules([
        (lambda x: x > 10, lambda x: x / 2)
    ])

    logic_system.export_rules("symbolic_rules.txt")
