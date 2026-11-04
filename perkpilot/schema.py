"""Pydantic models for every structured output in perkpilot.

Every extracted field is a `Sourced[T]`: the value plus the character span in the card's
terms document that supports it. Extraction rejects any field whose quote cannot be found.
"""
from __future__ import annotations

from datetime import date
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

Category = Literal[
    "dining", "us_supermarkets", "flights", "hotels", "travel", "drugstores", "gas", "other",
]
Channel = Literal["airline_direct", "hotel_direct", "amex_travel", "chase_travel"]
Period = Literal["monthly", "semi_annual", "annual"]
Reset = Literal["calendar_month", "calendar_half", "calendar_year", "cardmember_year"]


class Span(BaseModel):
    model_config = ConfigDict(frozen=True)
    clause_id: str
    start: int  # absolute character offsets into the card's terms document
    end: int


class Sourced(BaseModel, Generic[T]):
    value: T
    span: Span


class Cap(BaseModel):
    amount: float
    period: Literal["calendar_year", "calendar_month", "cardmember_year"]
    after_multiplier: float


class EarningRule(BaseModel):
    id: str
    card: str
    clause_id: str
    category: Sourced[Category]
    multiplier: Sourced[float]
    cap: Sourced[Cap] | None = None
    exclusions: list[Sourced[str]] = Field(default_factory=list)
    booking_channel: Sourced[list[Channel]] | None = None  # None means any channel


class Credit(BaseModel):
    id: str
    card: str
    clause_id: str
    name: Sourced[str]
    amount: Sourced[float]
    period: Sourced[Period]
    reset: Sourced[Reset] | None = None
    eligible_merchants: list[Sourced[str]] = Field(default_factory=list)
    enrollment_required: Sourced[bool] | None = None
    exclusions: list[Sourced[str]] = Field(default_factory=list)


class Fee(BaseModel):
    card: str
    annual_fee: Sourced[float] | None = None
    foreign_transaction_fee_pct: Sourced[float] | None = None


class CardTerms(BaseModel):
    card: str
    name: str
    earning_rules: list[EarningRule] = Field(default_factory=list)
    credits: list[Credit] = Field(default_factory=list)
    fees: Fee


# ---- raw LLM output: values with verbatim quotes, before span resolution ----

class RawField(BaseModel, Generic[T]):
    value: T
    quote: str


class RawEarningRule(BaseModel):
    category: RawField[Category]
    multiplier: RawField[float]
    cap: RawField[Cap] | None = None
    exclusions: list[RawField[str]] = Field(default_factory=list)
    booking_channel: RawField[list[Channel]] | None = None


class RawCredit(BaseModel):
    name: RawField[str]
    amount: RawField[float]
    period: RawField[Period]
    reset: RawField[Reset] | None = None
    eligible_merchants: list[RawField[str]] = Field(default_factory=list)
    enrollment_required: RawField[bool] | None = None
    exclusions: list[RawField[str]] = Field(default_factory=list)


class RawFees(BaseModel):
    annual_fee: RawField[float] | None = None
    foreign_transaction_fee_pct: RawField[float] | None = None


class RawClauseExtraction(BaseModel):
    earning_rules: list[RawEarningRule] = Field(default_factory=list)
    credits: list[RawCredit] = Field(default_factory=list)
    fees: RawFees = Field(default_factory=RawFees)


class Rejection(BaseModel):
    card: str
    clause_id: str
    field: str
    value: str
    quote: str
    reason: str


class ValidationVerdict(BaseModel):
    agrees: bool
    issues: list[str] = Field(default_factory=list)


class ValidationRecord(BaseModel):
    card: str
    rule_id: str
    kind: Literal["earning_rule", "credit", "fee"]
    agrees: bool
    issues: list[str]
    validator: str


# ---- transactions ----

PaymentMethod = Literal["card_present", "online", "mobile_card_reader", "third_party_processor", "digital_wallet"]


class Transaction(BaseModel):
    id: str
    date: date
    card: str
    merchant: str
    category: str
    amount: float
    payment_method: PaymentMethod = "card_present"
    foreign: bool = False


# ---- agent answers ----

class Citation(BaseModel):
    clause_id: str
    start: int | None = None  # absolute offsets of the highlighted span, if narrower than the clause
    end: int | None = None


class Claim(BaseModel):
    text: str
    citations: list[Citation]


class Recommendation(BaseModel):
    card: str
    expected_value: float
    points: float
    rationale: str
    citations: list[Citation] = Field(default_factory=list)


class Answer(BaseModel):
    answer: str
    abstained: bool = False
    value: str | float | bool | list[str] | None = None  # machine-readable answer, used by the eval
    claims: list[Claim] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    dropped_claims: list[str] = Field(default_factory=list)  # claims removed for lack of a valid citation
