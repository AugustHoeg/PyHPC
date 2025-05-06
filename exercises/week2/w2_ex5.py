import sys
import math

# Read command-line arguments (excluding script name)
components = sys.argv[1:]

# Convert arguments to floats
try:
    vector = [float(x) for x in components]
except ValueError:
    print("Please provide only numeric values.")
    sys.exit(1)

# Compute the magnitude (Euclidean norm)
magnitude = math.sqrt(sum(x**2 for x in vector))

# Print the result
print(magnitude)
