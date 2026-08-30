"""Structural tests for the reusable four-chart dashboard page."""

import unittest

from dashboard_workspace import DASHBOARD_GRAPH_COUNT, DASHBOARD_WORKSPACES
from layout import NAV_LINKS, create_layout


def walk_components(root):
    stack = [root]
    while stack:
        component = stack.pop()
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(reversed(children))
        elif children is not None and not isinstance(
            children, (str, int, float, bool)
        ):
            stack.append(children)


def component_by_id(root, component_id):
    return next(
        component
        for component in walk_components(root)
        if getattr(component, "id", None) == component_id
    )


class DashboardWorkspaceTests(unittest.TestCase):
    def test_dashboard_is_a_permanent_page_with_four_graphs(self):
        layout = create_layout()
        page = component_by_id(layout, "page-dashboard")
        ids = {
            getattr(component, "id", None)
            for component in walk_components(page)
        }

        self.assertEqual(len(DASHBOARD_WORKSPACES), DASHBOARD_GRAPH_COUNT)
        for workspace in DASHBOARD_WORKSPACES:
            self.assertIn(workspace.graph_id, ids)
            self.assertIn(workspace.paper_id, ids)
        self.assertIn({"label": "Дашборд", "href": "/dashboard"}, NAV_LINKS)

    def test_dashboard_graphs_have_disjoint_component_ids(self):
        layout = create_layout()
        page = component_by_id(layout, "page-dashboard")
        ids = [
            getattr(component, "id", None)
            for component in walk_components(page)
            if getattr(component, "id", None) is not None
        ]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            len([
                component
                for component in walk_components(page)
                if "dashboard-graph-cell" in (
                    getattr(component, "className", "") or ""
                ).split()
            ]),
            DASHBOARD_GRAPH_COUNT,
        )

    def test_each_graph_owns_independent_fields_and_settings(self):
        field_ids = [workspace.field_id("x") for workspace in DASHBOARD_WORKSPACES]
        theme_ids = [
            workspace.settings_control_id("theme")
            for workspace in DASHBOARD_WORKSPACES
        ]

        self.assertEqual(len(field_ids), len(set(field_ids)))
        self.assertEqual(len(theme_ids), len(set(theme_ids)))
        self.assertEqual(
            len({
                workspace.settings_id("font-size-legend")
                for workspace in DASHBOARD_WORKSPACES
            }),
            DASHBOARD_GRAPH_COUNT,
        )
        self.assertTrue(all(workspace.initial_height == 360 for workspace in DASHBOARD_WORKSPACES))


if __name__ == "__main__":
    unittest.main()
