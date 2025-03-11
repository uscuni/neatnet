# Contributing to `neatnet`

First off, thanks for taking the time to contribute! ❤️

All types of contributions are encouraged and valued. See this page for different ways to help and details about how this project handles them. Please make sure to read the relevant section before making your contribution. It will make it a lot easier for us maintainers and smooth out the experience for all involved. The community looks forward to your contributions. 🎉

> And if you like the project, but just don't have time to contribute, that's fine. There are other easy ways to support the project and show your appreciation, which we would also be very happy about:
> - Star the project
> - Tweet about it
> - Refer this project in your project's `README`
> - Mention the project at local meetups and tell your friends/colleagues

## I Have a Question

> If you want to ask a question, we assume that you have read the available [Documentation](https://uscuni.org/neatnet/).

Before you ask a question, it is best to search for existing [Issues](https://github.com/uscuni/neatnet/issues) that might help you. In case you have found a suitable issue and still need clarification, you can write your question in this issue. It is also advisable to search the internet for answers first, especially [Stack Overflow](https://stackoverflow.com).

If you then still feel the need to ask a question and need clarification, we recommend the following:

- Open an [Issue](https://github.com/uscuni/neatnet/issues/new).
- Provide as much context as you can about what you're running into.
- Provide project and platform versions (`python`, `shapely`, `geopandas`, etc.), depending on what seems relevant.

We will then take care of the issue as soon as possible.

## I Want To Contribute

### Reporting Bugs

#### Before Submitting a Bug Report

A good bug report shouldn't leave others needing to chase you up for more information. Therefore, we ask you to investigate carefully, collect information and describe the issue in detail in your report. Please complete the following steps in advance to help us fix any potential bug as fast as possible.

- Make sure that you are using the latest version.
- Determine if your bug is really a bug and not an error on your side, e.g. using incompatible environment components/versions (Make sure that you have read the [documentation](https://uscuni.org/neatnet/).).
- To see if other users have experienced (and potentially already solved) the same issue you are having, check if there is not already a bug report existing for your bug or error in the [bug tracker](https://github.com/uscuni/neatnet/issues).
- Also make sure to search the internet (especially [Stack Overflow](https://stackoverflow.com)) to see if users outside of the GitHub community have discussed the issue.
- Collect information about the bug:
  - Stack trace (Traceback)
  - OS, Platform and Version (Windows, Linux, macOS, x86, ARM)
  - Version of Python and relevant dependencies.
  - Possibly your input and the output
  - Can you reliably reproduce the issue? And can you also reproduce it with older versions?

#### How Do I Submit a Good Bug Report?

We use GitHub issues to track bugs and errors. If you run into an issue with the project:

- Open an [Issue](https://github.com/uscuni/neatnet/issues/new). (Since we can't be sure at this point whether it is a bug or not, we ask you not to talk about a bug yet and not to label the issue.)
- Explain the behavior you would expect and the actual behavior.
- Please provide as much context as possible and describe the *reproduction steps* that someone else can follow to recreate the issue on their own. This usually includes your code. For good bug reports you should isolate the problem and create a reduced test case. This is known as a [mininum reproducible example](https://en.wikipedia.org/wiki/Minimal_reproducible_example#:~:text=In%20computing%2C%20a%20minimal%20reproducible,to%20be%20demonstrated%20and%20reproduced.) – or MRE for short.
- Provide the information you collected in the previous section.

Once it's filed:

- The project team will label the issue accordingly.
- A team member will try to reproduce the issue with your provided steps. If there are no reproduction steps or no obvious way to reproduce the issue, the team will ask you for those steps.
- If the team is able to reproduce the issue, it will be left to be implemented by someone.

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for `neatnet`, **including completely new features and minor improvements to existing functionality**. Following these guidelines will help maintainers and the community to understand your suggestion and find related suggestions.

#### Before Submitting an Enhancement

- Make sure that you are using the latest version.
- Read the [documentation](https://uscuni.org/neatnet/) carefully and find out if the functionality is already covered, maybe by an individual configuration.
- Perform a [search](https://github.com/uscuni/neatnet/issues) to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.
- Find out whether your idea fits with the scope and aims of the project. It's up to you to make a strong case to convince the project's developers of the merits of this feature. Keep in mind that we want features that will be useful to the majority of our users and not just a small subset. If you're just targeting a minority of users, consider writing an add-on/plugin library.

#### How Do I Submit a Good Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://github.com/uscuni/neatnet/issues).

- Use a **clear and descriptive title** for the issue to identify the suggestion.
- Provide a **step-by-step description of the suggested enhancement** in as many details as possible.
- **Describe the current behavior** and **explain which behavior you expected to see instead** and why. At this point you can also tell which alternatives do not work for you.
- **Explain why this enhancement would be useful** to most neatnet's users. You may also want to point out the other projects that solved it better and which could serve as inspiration.

### Code Contribution

You can create a development environment using [Pixi](https://pixi.sh):

```sh
pixi install -e tests
```

Or with conda using the `environment.yml` file:

```sh
conda env create -f environment.yml
```

To install `neatnet` to the environment in an editable form, clone the repository, navigate to the main directory and install it with pip:

```sh
pip install -e .
```

When submitting a pull request:

- All existing tests should pass. Please make sure that the test suite passes, both locally and on GitHub Actions. Status on GHA will be visible on a pull request. GHA are automatically enabled on your own fork as well. To trigger a check, make a PR to your own fork.
- Ensure that documentation has built correctly. It will be automatically built for each PR.
- New functionality ***must*** include tests. Please write reasonable tests for your code and make sure that they pass on your pull request.
- Classes, methods, functions, etc. should have docstrings. The first line of a docstring should be a standalone summary. Parameters and return values should be documented explicitly.
- Follow PEP 8 when possible. We use ``Ruff`` for linting and formatting to ensure robustness & consistency in code throughout the project. It included in the ``pre-commit`` hook and will be checked on every PR.
- `neatnet` supports Python 3.11+ only. When possible, do not introduce additional dependencies. If that is necessary, make sure they can be treated as optional.

#### Procedure

1. *After* opening an issue and discussing with the development team, create a PR with the proposed changes.
2. If [testing fails](https://github.com/uscuni/neatnet/actions/runs/11368511561) due to an update in the code base:
3. Observed data is [saved as artifacts](https://github.com/uscuni/neatnet/actions/runs/11368511561#artifacts) from the workflow and can be download locally.
4. We determine the `ci_artifacts-ubuntu-latest-py313_latest` data as the "truth."
5. After comparison of the current "known" data with new data from (3.), if new data is "truthier," update your PR with the new "known" data.

#### Handling Edge Cases in Testing

Edge cases will crop up in full-scale FUA testing that we can ignore (following a thorough investigation – e.g. [`neatnet#77`](https://github.com/uscuni/neatnet/issues/77)) during testing. Once it is determined the geometry in question is not caused by a bug on our end, it can be added to the `KNOWN_BAD_GEOMS` collection in `tests/conftest.py`. This collection is a dictionary keyed by `<NAME>_CODE` of the city/FUA where the values are lists of index locations of simplified edges that can be ignored if they fail equality testing. As an example, see our initial "bad" geometries [here](https://github.com/uscuni/neatnet/blob/1be6b44b1a06d52453ecbaee205ae649101c4ea4/neatnet/tests/conftest.py#L25-L39), which were due to a variant number of coordinates in those resultant simplified edges created by [different versions of `shapely`](https://github.com/uscuni/neatnet/pull/67#issuecomment-2457333724).

##### Code Structure

Code should be linted and formatted via `ruff`. With the [`.pre-commit` hooks](https://github.com/uscuni/neatnet/blob/main/.pre-commit-config.yaml) we have code in commits will be formatted and linted automatically once [`pre-commit` is installed](https://pre-commit.com/#installation).

## Attribution

This guide is based on the **contributing-gen**. [Make your own](https://github.com/bttger/contributing-gen)!
