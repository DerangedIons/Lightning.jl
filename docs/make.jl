using Lightning
using Documenter

DocMeta.setdocmeta!(Lightning, :DocTestSetup, :(using Lightning); recursive=true)

makedocs(;
    modules=[Lightning],
    authors="Kyle Beggs (beggskw@gmail.com) and contributors",
    sitename="Lightning.jl",
    format=Documenter.HTML(;
        canonical="https://DerangedIons.github.io/Lightning.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/DerangedIons/Lightning.jl",
    devbranch="main",
)
