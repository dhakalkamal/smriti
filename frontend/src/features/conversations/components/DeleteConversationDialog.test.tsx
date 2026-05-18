import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { DeleteConversationDialog } from "./DeleteConversationDialog";

describe("DeleteConversationDialog", () => {
  it("displays the conversation title", () => {
    render(
      <DeleteConversationDialog
        errorMessage={null}
        isPending={false}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
        open={true}
        title="Daily notes"
      />,
    );

    expect(screen.getByRole("dialog", { name: "Delete conversation?" })).toHaveTextContent(
      "Daily notes",
    );
  });

  it("displays the fallback title", () => {
    render(
      <DeleteConversationDialog
        errorMessage={null}
        isPending={false}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
        open={true}
        title={null}
      />,
    );

    expect(screen.getByRole("dialog", { name: "Delete conversation?" })).toHaveTextContent(
      "Untitled conversation",
    );
  });

  it("closes on Cancel without confirming", () => {
    const confirm = vi.fn();

    render(<DialogHarness onConfirm={confirm} />);

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(confirm).not.toHaveBeenCalled();
  });

  it("closes on Escape without confirming", () => {
    const confirm = vi.fn();

    render(<DialogHarness onConfirm={confirm} />);

    fireEvent.keyDown(window, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(confirm).not.toHaveBeenCalled();
  });

  it("disables Cancel and Delete while pending", () => {
    render(
      <DeleteConversationDialog
        errorMessage={null}
        isPending={true}
        onCancel={vi.fn()}
        onConfirm={vi.fn()}
        open={true}
        title="Daily notes"
      />,
    );

    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Delete" })).toBeDisabled();
  });
});

function DialogHarness({ onConfirm }: { onConfirm: () => void }) {
  const [open, setOpen] = useState(true);

  return (
    <DeleteConversationDialog
      errorMessage={null}
      isPending={false}
      onCancel={() => {
        setOpen(false);
      }}
      onConfirm={onConfirm}
      open={open}
      title="Daily notes"
    />
  );
}
