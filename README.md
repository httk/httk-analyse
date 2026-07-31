# httk-[placeholder]

*httk-[placeholder]* is a [*httk₂*](https://github.com/httk/httk2) module providing [placeholder functionality].

## Registry

When the module exposes a loader, adapter, command, provider, record, or
schema, add an import-light shim under the matching `httk.registry` tier. The
template keeps the example here rather than adding an inactive registry
package; replace `placeholder` with the module's existing namespace name:

```python
# src/httk/registry/placeholder/__init__.py
# from httk.core.register import register_loader
# register_loader(name="placeholder", loader="httk.placeholder.io:load", extensions=(".placeholder",))

# src/httk/registry/cli/placeholder/__init__.py
# from httk.core import register_cli_command
# register_cli_command("placeholder", "httk.placeholder.cli:command", "run placeholder")

# src/httk/registry/entries/placeholder/__init__.py
# from httk.core import register_entry_provider, register_entry_record
# register_entry_provider(name="placeholder", factory="httk.placeholder.entries:Provider")
# register_entry_record(name="placeholder-record", record="httk.placeholder.records:Record")

# src/httk/registry/schemas/placeholder/__init__.py
# from httk.core import register_entry_type_schema, register_property_definition
# register_entry_type_schema(
#     definition_id="https://schemas.example.org/defs/v1/entrytypes/placeholder",
#     resource="httk.registry.schemas.placeholder:placeholder.json",
# )
# register_property_definition(
#     definition_id="https://schemas.example.org/defs/v1/properties/placeholder/value",
#     resource="httk.registry.schemas.placeholder:value.json",
# )
```

Keep these shims limited to registration calls and lazy `"module:attribute"`
references; importing `httk.core` discovers them without importing the module
implementation. See the [httk-core registry guide](https://docs.httk.org/httk-core/registry.html)
for the tiers, identity rules, and module layout.
