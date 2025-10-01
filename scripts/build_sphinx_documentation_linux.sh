# Copy dependencies
cd ..
cp -R _static docs/source/
cp -R examples docs/source/

# Delete old kcpdi modules

rm -f docs/source/kcpdi*.rst

# Generate package docstring

sphinx-apidoc -o docs/source kcpdi

# Generate HTML

cd docs
make clean
make html

# Clean temp directories
rm -Rf source/_static
rm -Rf source/examples
