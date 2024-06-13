The project structure will evolve as needed, but let's start from something like

```
.
├── README.md       
├── LICENSE
├── checkpoints     <-- checkpoints and models
├── data
│   ├── processed   <-- data ready to be loaded/analyzed
│   └── raw         <-- immutable unprocessed data
├── notebooks       <-- notebooks, figures, reports
├── references      <-- data dictionaries, dataset docs, any other useful docs
└── src             <-- code, tests, 
```

Please do not check any data into git.

Use the notebooks folder for exploratory stuff and reports (including figures, presentations, markdown).
