using OkMLModels
using Documenter

DocMeta.setdocmeta!(OkMLModels, :DocTestSetup, :(using OkMLModels); recursive=true)

makedocs(;
    modules=[OkMLModels],
    authors="okatsn <okatsn@gmail.com> and contributors",
    repo="https://github.com/okatsn/OkMLModels.jl/blob/{commit}{path}#{line}",
    sitename="OkMLModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://okatsn.github.io/OkMLModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/okatsn/OkMLModels.jl",
    devbranch="main",
)
