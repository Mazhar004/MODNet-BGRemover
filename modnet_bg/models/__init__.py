"""Vendored MODNet network, unchanged from upstream.

Upstream shipped this directory without an __init__.py and relied on implicit
namespace packages. That works for a source checkout but makes setuptools'
package discovery skip it, so the wheel would omit the network entirely.
"""
