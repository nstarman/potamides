# Using Matplotlib in Documentation

This project's documentation supports embedding matplotlib figures directly in
the documentation using the `plot` directive.

## Method 1: Plot Directive (Recommended)

Use the `plot` directive in your Markdown files to generate figures
automatically during the documentation build:

````markdown
```{plot}
:context: close-figs

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(6, 4))
plt.plot(x, y, 'b-', linewidth=2)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave')
plt.grid(True, alpha=0.3)
```
````

### Plot Directive Options

- `:context:` - Preserve variables between plot blocks
- `:context: close-figs` - Close previous figures before plotting
- `:include-source:` - Show/hide source code (default: True)
- `:nofigs:` - Execute code but don't generate figures

## Method 2: Code Blocks with Plots

You can also show matplotlib code examples in regular Python code blocks:

````markdown
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.show()
```
````

## Method 3: Static Images

For pre-generated images, use standard Markdown syntax:

```markdown
![Description](path/to/image.png)
```

## Configuration

The matplotlib plot directive is configured in `docs/conf.py`:

<!-- blacken-docs:off -->

    extensions = [
        ...
        "matplotlib.sphinxext.plot_directive",
    ]

    plot_include_source = True
    plot_html_show_source_link = False
    plot_html_show_formats = False
    plot_formats = ["png"]
    plot_rcparams = {
        "figure.figsize": (8, 6),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
    }

<!-- blacken-docs:on -->

## Best Practices

1. **Use `:context: close-figs`** to prevent figure accumulation
2. **Set appropriate figure sizes** using `plt.figure(figsize=(w, h))`
3. **Add labels and titles** to make plots self-explanatory
4. **Use grid and legends** to improve readability
5. **Close figures explicitly** when needed with `plt.close()`

## Example: Complete Plot

````markdown
```{plot}
:context: close-figs

import numpy as np
import matplotlib.pyplot as plt

# Generate data
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + 0.5 * np.sin(3*theta)
x = r * np.cos(theta)
y = r * np.sin(theta)

# Create plot
fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
ax.plot(x, y, 'b-', linewidth=2, label='Polar curve')
ax.plot(0, 0, 'ro', markersize=8, label='Origin')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Rose Curve: r = 1 + 0.5·sin(3θ)')
ax.grid(True, alpha=0.3)
ax.axis('equal')
ax.legend()
plt.tight_layout()
```
````

This will automatically generate and embed the plot in your documentation when
built with Sphinx/ReadTheDocs.
