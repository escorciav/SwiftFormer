# Visualization

I'm a visual person.
My brain demands pictoric visual representations for understanding 🧠.
As such, I value the communication visual stories.

## [Vega and Vega Lite](https://vega.github.io/editor)

Quick and human friendly-grammar for visualization.

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "description": "A simple bar chart with rounded corners at the end of the bar.",
  "data": {
    "values": [
        {"Op": "Post Norm", "Latency (%)": 48.54},
{"Op": "MLP", "Latency (%)": 14.58},
{"Op": "ROPE", "Latency (%)": 8.25},
{"Op": "Att Mask", "Latency (%)": 6.52},
{"Op": "Softmax", "Latency (%)": 5.49},
{"Op": "Att KQ", "Latency (%)": 4.84},
{"Op": "Att AV", "Latency (%)": 4.01},
{"Op": "Input Norm", "Latency (%)": 1.66},
{"Op": "Q", "Latency (%)": 1.47},
{"Op": "K", "Latency (%)": 1.43},
{"Op": "V", "Latency (%)": 1.37},
{"Op": "Add", "Latency (%)": 1.15},
{"Op": "others", "Latency (%)": 0.51},
{"Op": "O", "Latency (%)": 0.17},
    ]
  },
  "mark": {"type": "bar", "cornerRadiusEnd": 4},
  "encoding": {
    "x": {"field": "Latency (%)", "type": "quantitative", "scale": {"domain": [0, 60]}},
    "y": {"field": "Op", "type": "ordinal", "sort": "-Latency (%)", "axis": {"title": null}},
    "color": {"field": "Op", "type": "nominal", "scale": {"scheme": "category20"}}
  },
  "layer": [
    {
      "mark": {"type": "bar", "cornerRadiusEnd": 4},
      "encoding": {
        "opacity": {"value": 1}
      }
    },
    {
      "mark": {
        "type": "text",
        "align": "left",
        "baseline": "middle",
        "dx": 5,
        "fontSize": 11
      },
      "encoding": {
        "text": {"field": "Latency (%)", "type": "quantitative", "format": ".2f"},
        "color": {"value": "black"}
      }
    }
  ]
}
```
